# Importing required libraries
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid')
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from pvlib import solarposition, location
from sklearn.linear_model import LinearRegression
import mysql.connector
# from pandas.plotting import register_matplotlib_converters
from datetime import date, datetime, timedelta
from PIL import Image
import warnings
import base64
import streamlit_authenticator as stauth
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from folium.plugins import Fullscreen
# Fullscreen().add_to(map)

warnings.filterwarnings('ignore')

st.set_page_config(
        page_title="HelioCloud- The Dashboard",
        page_icon='SmartHelio logo favicon.png',
        layout="wide",
    )
# st.components.v1.html(html, width=None, height=0, scrolling=False)
hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
#st.set_page_config(layout="wide")



@st.cache
def connecting_to_server(sensor_ID):
    #estabilishing connection with the Mysql server
    db_connection = mysql.connector.connect(host="helio-smsapi-db.ledsafari.com",
        user="root", passwd="helio#123", database="helio")
    my_database = db_connection.cursor()
    sql ="""SELECT * FROM T_SENSOR_STATISTICS where sensorId = \"""" + sensor_ID + """\""""
    my_database.execute(sql)
    output = my_database.fetchall()
    server_data = pd.DataFrame(output, columns=['sensor_ID', 'datetime', 'Mod. Temp. (°C)', 'Current(I)', 'Voltage(V)'])
    server_data = server_data.fillna(value={'Voltage(V)': 0})
    server_data = server_data[server_data['datetime'] > 0]
    server_data = server_data.sort_values(by=['sensor_ID', 'datetime'])
    m_datetime = server_data['datetime']
    # Convert to datetime format
    m_datetime = [datetime.utcfromtimestamp(x) for x in m_datetime]
    server_data['Voltage(V)'] = server_data['Voltage(V)']
    server_data['Current(I)'] = server_data['Current(I)'] / 1000
    server_data['Power(W)'] = server_data['Current(I)'] * server_data['Voltage(V)']
    server_data['datetime'] = m_datetime
    sensor_data = server_data
    i = (sensor_data['Current(I)'] > 0) & (sensor_data['Voltage(V)'] > 0)
    sensor_data= sensor_data[i]
    return sensor_data

def energy_yield(input_data):
    input_power = input_data.copy()
    input_datetime = input_data.reset_index()
    input_datetime = pd.DataFrame(
        input_datetime['datetime'], columns=['datetime'])
    input_datetime['datetime'] = pd.to_datetime(input_datetime['datetime'])
    b = (input_datetime - input_datetime.iloc[0]) / timedelta(hours=1)
    b = b.diff()
    step = float(b.mode().values)

    # Flag with NaN the beginning of the days
    b = b.where(b < 6)
    input_datetime.loc[:, 'timeinterval'] = b.values
    input_datetime = input_datetime.set_index('datetime')
    # print(input_datetime)

    # Calculate the energy generated for each row
    output_energy = (input_power + input_power.shift(periods=1, fill_value=0)).mul(
        input_datetime['timeinterval'].div(2), axis=0)

    # First hour of the day (considering half step of rise before first measurment)
    i = output_energy.isna()
    output_energy[i] = input_power[i] * step / 2

    # last hour of the days (considering half step of decay after last measurment)
    ishift = i.shift(periods=-1, fill_value=True)
    output_energy[ishift] = output_energy[ishift] + \
        input_power[ishift] * step / 2
    # print(output_energy)
    output_energy.index = pd.to_datetime(output_energy.index)
    # Aggregation by day
    daily_yield = output_energy.groupby(pd.Grouper(freq='D')).sum()

    total_yield = daily_yield.sum(axis=0, skipna=True)
    # daily_yield.columns = ['Daily yield']
    

    return daily_yield, total_yield, output_energy

def main(sensor_ID, start_date, end_date, sensor_data):
    #st.write(end_date)
    #filtering data based on sensor ID
    sensor_data = sensor_data[sensor_data['sensor_ID'] == sensor_ID]
    end_date = pd.to_datetime(end_date) + pd.DateOffset(days=1)
    #filtering data based on start date and end date
    sensor_data_final = sensor_data.loc[(sensor_data['datetime'] > start_date) & ((sensor_data['datetime'] <= end_date))]
    sensor_data_final.set_index('datetime', inplace = True)
    sensor_data_final.drop(columns=['sensor_ID'], inplace = True)
    #st.write(end_date)
    return sensor_data_final
def sub_main(sensor_ID, sensor_data):
    #filtering data based on sensor ID
    sensor_data_final = sensor_data[sensor_data['sensor_ID'] == sensor_ID]
    sensor_data_final.set_index('datetime', inplace = True)
    sensor_data_final.drop(columns=['sensor_ID'], inplace = True)
    
    return sensor_data_final
    
    
def get_tcoef(df_in, param_column, temp_column):
  df = df_in.copy()
  df.dropna(inplace=True)
  # get x and y
  X = df[[temp_column]]
  y = df[[param_column]]
  # linear regression
  regressor = LinearRegression()
  lr = regressor.fit(X, y)
  # get y predicted
  y_pred = lr.predict(X)
  pred = pd.DataFrame(y_pred, index=X.index, columns=['y_pred'])
  # get parameters
  coef = lr.coef_
  inter = lr.intercept_

  return pred, coef, inter


# filter function
def filter_reg(df_in, param_column, temp_column, q1=0.25, q3=0.75):
  df = df_in.copy()
  df.dropna(inplace=True)
  # Filter by param
  Q1 = df[param_column].quantile(q1)
  Q3 = df[param_column].quantile(q3)
  IQR = Q3 - Q1
  df["col_filtered"] = df[param_column][~((df[param_column] < (Q1 - 1.5 * IQR)) | (df[param_column] > (Q3 + 1.5 * IQR)))]
  # Filter by Temp
  Q1 = df[temp_column].quantile(q1)
  Q3 = df[temp_column].quantile(q3)
  IQR = Q3 - Q1
  df["col_filtered"] = df.col_filtered[~((df[temp_column] < (Q1 - 1.5 * IQR)) | (df[temp_column] > (Q3 + 1.5 * IQR)))]
  df["pred"], a, b = get_tcoef(df_in=df,
                                  param_column="col_filtered",
                                  temp_column=temp_column)
  df["scores"] = df[param_column] - df.pred
  # Filter by meaninful scores
  Q1 = df.scores.quantile(q1)
  Q3 = df.scores.quantile(q3)
  IQR = Q3 - Q1

  df["ret_col"] = df.col_filtered[~((df.scores < (Q1 - 1.5 * IQR)) | (df.scores > (Q3 + 1.5 * IQR)))]

  return df.ret_col
  
  
# with open('style.css') as f:
    # st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


######################################################################################################################

#smarthelio LOGO

st.sidebar.image('SmartHelio logo (335 x 100 px)-2.png', width = 200)

File_object = pd.read_csv("user_data.csv", usecols = ['usernames','passwords','names','CCID']) 
File_object1 = pd.read_csv("sensors.csv") 


hashed_passwords = stauth.hasher(File_object['passwords']).generate()
authenticator = stauth.authenticate(File_object['names'],File_object['usernames'],hashed_passwords,'some_cookie_name','some_signature_key',cookie_expiry_days=0)

name, authentication_status = authenticator.login('Log on to Dashboard','sidebar')

File_object.set_index('names', inplace = True)
# st.write(File_object.at[name, 'CCID'])

#st.write(CCID)
if authentication_status:
    st.sidebar.write('Welcome User : *%s*' % (name))
    CCID = [i for i in  File_object.at[name, 'CCID'].split(';')]
    # sensor_data = connecting_to_server()
    sensor_ID = st.sidebar.selectbox('Device ID', CCID)
    sensor_data = connecting_to_server(sensor_ID)
    # st.write(sensor_data)
    option = st.sidebar.selectbox('Select Analysis',('DC Power Time Series', 'Energy Production', 'Current Voltage Time Series', 'Temperature Coefficient of Voltage', 'Degradation Analysis of Current and Voltage'))
    start_date = st.sidebar.date_input("Enter Start Date",date.today())
    end_date = st.sidebar.date_input("Enter End Date",date.today())
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    html = """
      <style>

        /* SH logo */
        .element-container:nth-child(1) {
          left: -0px;
          top:  -80px;
        }
        .element-container:nth-child(2) {
          left: -0px;
          top:  385px;
        }
        .element-container:nth-child(3) {
          left: -0px;
          top:  -140px;
        }
        .element-container:nth-child(4) {
          left: -0px;
          top:  -140px;
        }
        .element-container:nth-child(5) {
          left: -0px;
          top:  -140px;
        }
        .element-container:nth-child(6) {
          left: -0px;
          top:  -140px;
        }
        .element-container:nth-child(7) {
          left: -0px;
          top:  -140px;
        }
        .element-container:nth-child(8) {
          left: -0px;
          top:  -140px;
        }
      </style>
    """
    # if(st.button("About")):
        # st.text("The dashboard displays instantaneous information recieved from IoT sensor & PV system characteritics")
        

    st.markdown(html, unsafe_allow_html=True)
    if sensor_ID not in list(sensor_data.sensor_ID):
        st.error("Please Enter Correct Device ID")
    else: 
       # st.text(sensor_ID)
        
        final_data1 = main(sensor_ID, start_date, end_date, sensor_data)
        daily_yield, total_yield, output_energy = energy_yield(final_data1['Power(W)'])
        final_data = pd.concat([final_data1,output_energy], axis = 1)
        final_data.columns = ['Mod Temp(°C)','Current(A)','Voltage(V)','Power(W)', 'Energy Yield(kWh)']
        final_data['Time(hhmm)'] = [datetime.strftime(d, "%H%M") for d in final_data.index]
        final_data['Time(hhmm)'] = final_data['Time(hhmm)'].astype(int)
        final_data['Date'] = final_data.index.date
        
        if len(final_data) != 0:
            
            final_data.reset_index(inplace = True)
    
            if timedelta(days=31) > (end_date - start_date) >= timedelta(days=1) and option == 'Energy Production':
               dff = final_data[['datetime', 'Energy Yield(kWh)']].copy()
               dff['Energy Yield(kWh)'] = dff['Energy Yield(kWh)']/1000
               dff.set_index('datetime', inplace = True)
               dff = dff.groupby(pd.Grouper(freq='d')).sum()
               dff.reset_index(inplace = True)
               fig = px.bar(        
                        dff,
                        x = "datetime",
                        y = "Energy Yield(kWh)", height=600, labels={"datetime": "Date","Energy Yield(kWh)": "Energy Yield (kWh)"},
                        title = "Energy Generated by PV module<br><sup>Calculated as product of Power and Time</sup>"
                    )
               fig.update_layout(title_x=0.5, title_y=0.9)
               fig['layout']['title']['font'] = dict(size=20)
               fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               # fig.update_xaxes(tickformat="%d-%m-%y")
               st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
               
            elif timedelta(days=90) >= (end_date - start_date) >= timedelta(days=31) and option == 'Energy Production':
               dff = final_data[['datetime', 'Energy Yield(kWh)']].copy()
               dff['Energy Yield(kWh)'] = dff['Energy Yield(kWh)']/1000
               dff.set_index('datetime', inplace = True)
               dff = dff.groupby(pd.Grouper(freq='W')).sum()
               dff.reset_index(inplace = True)
               fig = px.bar(        
                        dff,
                        x = "datetime",
                        y = "Energy Yield(kWh)", height=600, labels={"datetime": "Weeks","Energy Yield(kWh)": "Energy Yield (kWh)"},
                        title = "Energy Generated by PV module<br><sup>Calculated as product of Power and Time</sup>"
                    )
               fig.update_layout(title_x=0.5, title_y=0.9)
               fig['layout']['title']['font'] = dict(size=20)
               fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               # fig.update_xaxes(tickformat="%d-%m-%y")
               st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
               
            elif (end_date - start_date) == timedelta(days=0) and option == 'Energy Production':
               dff = final_data[['datetime', 'Energy Yield(kWh)']].copy()
               dff['Energy Yield(kWh)'] = dff['Energy Yield(kWh)']/1000
               dff.set_index('datetime', inplace = True)
               dff = dff.groupby(pd.Grouper(freq='H')).sum()
               dff.reset_index(inplace = True)
               fig = px.bar(        
                        dff,
                        x = "datetime",
                        y = "Energy Yield(kWh)", height=600, labels={"datetime": "Time","Energy Yield(kWh)": "Energy Yield (kWh)"},
                        title = "Energy Generated by PV module<br><sup>Calculated as product of Power and Time</sup>"
                    )
               fig.update_layout(title_x=0.5, title_y=0.9)
               fig['layout']['title']['font'] = dict(size=20)
               fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
               
            elif (end_date - start_date) > timedelta(days=90) and option == 'Energy Production':
               dff = final_data[['datetime', 'Energy Yield(kWh)']].copy()
               dff['Energy Yield(kWh)'] = dff['Energy Yield(kWh)']/1000
               dff.set_index('datetime', inplace = True)
               dff = dff.groupby(pd.Grouper(freq='M')).sum()
               dff.reset_index(inplace = True)
               fig = px.bar(        
                        dff,
                        x = "datetime",
                        y = "Energy Yield(kWh)", height=600, labels={"datetime": "Months","Energy Yield(kWh)": "Energy Yield (kWh)"},
                        title = "Energy Generated by PV module<br><sup>Calculated as product of Power and Time</sup>"
                    )
               fig.update_layout(title_x=0.5, title_y=0.9)
               fig['layout']['title']['font'] = dict(size=20)
               fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})   
                 
            elif timedelta(days=31) > (end_date - start_date) > timedelta(days=1) and option == 'DC Power Time Series':
               dff = final_data[['datetime', 'Power(W)']].copy()
               
               fig = px.line(        
                        dff,
                        x = "datetime",
                        y = "Power(W)", height=600, labels={"datetime": "Date","Power(W)": "Power (W)"},
                        title = "DC Power of PV Module<br><sup>Calculated as product of Current and Voltage</sup>"
                    )
               fig.update_layout(title_x=0.5, title_y=0.9)
               fig['layout']['title']['font'] = dict(size=20)
               fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
               fig.update_xaxes(rangeslider_thickness = 0.05)
               st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
               
            elif (end_date - start_date) >= timedelta(days=31) and option == 'DC Power Time Series':
               dff = final_data[['datetime', 'Power(W)']].copy()
               
               fig = px.line(        
                        dff,
                        x = "datetime",
                        y = "Power(W)", height=600, labels={"datetime": "Date","Power(W)": "Power (W)"},
                        title = "DC Power of PV Module<br><sup>Calculated as product of Current and Voltage</sup>"
                    )
               fig.update_layout(title_x=0.5, title_y=0.9)
               fig['layout']['title']['font'] = dict(size=20)
               fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
               fig.update_xaxes(rangeslider_thickness = 0.05)
               st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
               
            elif (end_date - start_date) == timedelta(days=1) and option == 'DC Power Time Series':
               dff = final_data[['datetime', 'Power(W)']].copy()
               
               fig = px.line(        
                        dff,
                        x = "datetime",
                        y = "Power(W)", height=600, labels={"datetime": "Time","Power(W)": "Power (W)"},
                        title = "DC Power of PV Module<br><sup>Calculated as product of Current and Voltage</sup>"
                    )
               fig.update_layout(title_x=0.5, title_y=0.9)
               fig['layout']['title']['font'] = dict(size=20)
               fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
               fig.update_xaxes(rangeslider_thickness = 0.05)
               st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
               
            elif (end_date - start_date) == timedelta(days=0) and option == 'DC Power Time Series':  
               dff = final_data[['datetime', 'Power(W)']].copy()
               
               fig = px.line(        
                        dff,
                        x = "datetime",
                        y = "Power(W)", height=600, labels={"datetime": "Time","Power(W)": "Power (W)"},
                        title = "DC Power of PV Module<br><sup>Calculated as product of Current and Voltage</sup>"
                    )
               fig.update_layout(title_x=0.5, title_y=0.9)
               fig['layout']['title']['font'] = dict(size=20)
               fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
               fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
               fig.update_xaxes(rangeslider_thickness = 0.05)
               st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
               
            elif timedelta(days=31) > (end_date - start_date) > timedelta(days=1) and option == 'Current Voltage Time Series':
                dff = final_data[['datetime','Current(A)', 'Voltage(V)']].copy()              
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=dff["datetime"],y=dff["Current(A)"], mode ="lines", name="Current"),secondary_y=False,)
                fig.add_trace(go.Scatter(x=dff["datetime"],y=dff["Voltage(V)"], mode ="lines", name="Voltage"),secondary_y=True,)
                # fig = px.line(        
                        # dff,
                        # x = "datetime",
                        # y = ["Current(A)","Voltage(V)"], labels={"datetime": "Time"}, height=600,
                        # title = "Current Voltage Time Series"
                    # )
                fig.update_layout(title= "PV Module I-V Time Series<br><sup>Electrical Parameters Measured by Device</sup>", title_x=0.45,title_y=0.9)
                fig.update_yaxes(range = [0,30], title_text= "Current (A)", secondary_y=False)
                fig.update_yaxes(range = [0,50], title_text= "Voltage (V)", secondary_y=True)
                fig.update_xaxes(title_text= "Date")
                fig['layout']['title']['font'] = dict(size=20)
                fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                #fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.95))
                fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                fig.update_xaxes(rangeslider_thickness = 0.05)
                fig.update_layout(height = 600)
                st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
                
            elif (end_date - start_date) >= timedelta(days=31) and option == 'Current Voltage Time Series':
                dff = final_data[['datetime','Current(A)', 'Voltage(V)']].copy()
                
               
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=dff["datetime"],y=dff["Current(A)"], mode ="lines", name="Current"),secondary_y=False,)
                fig.add_trace(go.Scatter(x=dff["datetime"],y=dff["Voltage(V)"], mode ="lines", name="Voltage"),secondary_y=True,)
                # fig = px.line(        
                        # dff,
                        # x = "datetime",
                        # y = ["Current(A)","Voltage(V)"], labels={"datetime": "Time"}, height=600,
                        # title = "Current Voltage Time Series"
                    # )
                fig.update_layout(title= "PV Module I-V Time Series<br><sup>Electrical Parameters Measured by Device</sup>", title_x=0.45,title_y=0.9)
                fig.update_yaxes(range = [0,30], title_text= "Current (A)", secondary_y=False)
                fig.update_yaxes(range = [0,50], title_text= "Voltage (V)", secondary_y=True)
                fig.update_xaxes(title_text= "Date")
                fig['layout']['title']['font'] = dict(size=20)
                fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                #fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.95))
                fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                fig.update_xaxes(rangeslider_thickness = 0.05)
                fig.update_layout(height = 600)
                st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
                
            elif (end_date - start_date) == timedelta(days=1) and option == 'Current Voltage Time Series':
                dff = final_data[['datetime','Current(A)', 'Voltage(V)']].copy() 
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=dff["datetime"],y=dff["Current(A)"], mode ="lines", name="Current"),secondary_y=False,)
                fig.add_trace(go.Scatter(x=dff["datetime"],y=dff["Voltage(V)"], mode ="lines", name="Voltage"),secondary_y=True,)
                # fig = px.line(        
                        # dff,
                        # x = "datetime",
                        # y = ["Current(A)","Voltage(V)"], labels={"datetime": "Time"}, height=600,
                        # title = "Current Voltage Time Series"
                    # )
                fig.update_layout(title= "PV Module I-V Time Series<br><sup>Electrical Parameters Measured by Device</sup>", title_x=0.45,title_y=0.9)
                fig.update_yaxes(range = [0,30], title_text= "Current (A)", secondary_y=False)
                fig.update_yaxes(range = [0,50], title_text= "Voltage (V)", secondary_y=True)
                fig.update_xaxes(title_text= "Time")
                fig['layout']['title']['font'] = dict(size=20)
                fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                #fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.95))
                fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                fig.update_xaxes(rangeslider_thickness = 0.05)
                fig.update_layout(height = 600)
                st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
                
            elif (end_date - start_date) == timedelta(days=0) and option == 'Current Voltage Time Series':  
                dff = final_data[['datetime','Current(A)', 'Voltage(V)']].copy()              
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=dff["datetime"],y=dff["Current(A)"], mode ="lines", name="Current"),secondary_y=False,)
                fig.add_trace(go.Scatter(x=dff["datetime"],y=dff["Voltage(V)"], mode ="lines", name="Voltage"),secondary_y=True,)
                # fig = px.line(        
                        # dff,
                        # x = "datetime",
                        # y = ["Current(A)","Voltage(V)"], labels={"datetime": "Time"}, height=600,
                        # title = "Current Voltage Time Series"
                    # )
                fig.update_layout(title= "PV Module I-V Time Series<br><sup>Electrical Parameters Measured by Device</sup>", title_x=0.45,title_y=0.9)
                fig.update_yaxes(range = [0,30], title_text= "Current (A)", secondary_y=False)
                fig.update_yaxes(range = [0,50], title_text= "Voltage (V)", secondary_y=True)
                
                fig.update_xaxes(title_text= "Time")
                fig['layout']['title']['font'] = dict(size=20)
                fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                #fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.95))
                fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                fig.update_xaxes(rangeslider_thickness = 0.05)
                fig.update_layout(height = 600)
                st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
                
            elif option == 'Temperature Coefficient of Voltage':
                sensor_data1 = connecting_to_server(sensor_ID)
                sensor_data_com = sub_main(sensor_ID, sensor_data1)
                sensor_data_com.rename(columns={'Mod. Temp. (°C)':'Tmod', 'Voltage(V)':'V', 'Current(I)':'I', 'Power(W)':'P'}, inplace = True)
#                 st.write(sensor_data_comp)
#                 st.write(sensor_data_comp.columns)
                Inputs = File_object1.loc[File_object1['sensor_CCID']==sensor_ID]
                # st.write(Inputs)
                alpha = Inputs['alpha'].tolist()[0]
                Isc = Inputs['I_sc'].tolist()[0]
                Impp = Inputs['I_mpp'].tolist()[0]
                Vmpp = Inputs['V_mpp'].tolist()[0]
                irr_lim_up = 1050 
                irr_lim_down = 950 
                lat = Inputs['latitude'].tolist()[0]
                lon = Inputs['longitude'].tolist()[0]
                alt = Inputs['altitude'].tolist()[0]
                tz = Inputs['time_zone'].tolist()[0]
                c_voltage = Inputs['beta'].tolist()[0]
                
                site = location.Location(lat, lon, tz=tz, altitude=alt, name=None)
                
                solpos = solarposition.get_solarposition(time=sensor_data_com.index,
                                           latitude=lat,
                                           longitude=lon)
                
                AM = site.get_airmass(times=None, solar_position=solpos, model='kastenyoung1989')
#                 st.write(AM)
                
                sensor_data_com['G'] = 1000*(1 + ((sensor_data_com['I'] - Impp)- (alpha/100)*(sensor_data_com['Tmod'] - 25))/Isc)
#                 st.write(sensor_data_comp.dropna().describe())
                sensor_data_com['V_norm'] = sensor_data_com['V']*100/Vmpp
                v_df = sensor_data_com.copy()
                AM_filter = AM[(AM['airmass_absolute']>=1.4) & (AM['airmass_absolute']<=1.6)]
                v_filter = v_df[v_df.index.isin(AM_filter.index)]
                v_filter["V_filter"] = v_filter['V_norm'].loc[(v_filter['G'] > irr_lim_down) & (v_filter['G'] < irr_lim_up)]
                
                # Filtering outliers
                v_filter["V_coef"] = filter_reg(df_in=v_filter,
                                                   param_column="V_filter",
                                                   temp_column='Tmod')
                
                # get coef
                v_filter["V_reg"], v_coef, v_inter = get_tcoef(df_in=v_filter,
                                                                  param_column="V_coef",
                                                                  temp_column="Tmod")
#                 st.write(v_coef, v_inter)

                fig = make_subplots()
                fig.add_trace(go.Scatter(x=v_filter["Tmod"],y=v_filter["V_norm"],name="All Values", mode ="markers", marker={'symbol': 'x', 'size': 2}))
                fig.add_trace(go.Scatter(x=v_filter["Tmod"],y=v_filter["V_coef"],name="Filtered Values", mode ="markers", marker={'size': 6}))
                fig.update_layout(title= "Variation of Voltage with Temperature<br><sup>Temperature coefficient of Voltage = " + str(np.round(v_coef.tolist()[0],2)) + " % per °C</sup>", title_x=0.5,title_y=0.95)
                fig.update_yaxes(range = [50,100], title_text= "Normalized Vmpp (V)")
                fig.update_xaxes(range = [10,70], title_text= "Module Temperature")
                fig['layout']['title']['font'] = dict(size=20)
                fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                fig.update_layout(height = 600)
                st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})
            
            elif option == 'Degradation Analysis of Current and Voltage':
                sensor_data1 = connecting_to_server(sensor_ID)
                sensor_data_com = sub_main(sensor_ID, sensor_data1)
                sensor_data_com.rename(columns={'Mod. Temp. (°C)':'Tmod', 'Voltage(V)':'V', 'Current(I)':'I', 'Power(W)':'P'}, inplace = True)
#                 st.write(sensor_data_comp)
#                 st.write(sensor_data_comp.columns)
                Inputs = File_object1.loc[File_object1['sensor_CCID']==sensor_ID]
                # st.write(Inputs)
                alpha = Inputs['alpha'].tolist()[0]
                Isc = Inputs['I_sc'].tolist()[0]
                Impp = Inputs['I_mpp'].tolist()[0]
                Vmpp = Inputs['V_mpp'].tolist()[0]
                
                sensor_data_com['G'] = 1000*(1 + ((sensor_data_com['I'] - Impp)- (alpha/100)*(sensor_data_com['Tmod'] - 25))/Isc)
                
                mfoc = sensor_data_com.copy()
#                 st.write(mfoc)
                mfoc['I_norm'] = mfoc['I']/Impp 
                mfoc['V_norm'] = mfoc['V']/Vmpp 
                mfoc['month'] = mfoc.index.month
                temp_df = mfoc[((mfoc['G']>790) & (mfoc['G']<810)) & ((mfoc['Tmod']>53) & (mfoc['Tmod']<57))].copy()
                
                trace1 = go.Box(y=temp_df['I_norm'], x=temp_df['month'], name='Current',boxpoints=False,)
                trace2 = go.Box(y=temp_df['V_norm'], x=temp_df['month'], name='Voltage',boxpoints=False,)
                data = [trace1, trace2]
                layout = go.Layout(yaxis=go.layout.YAxis(title='Normalized Value', range=[0.6, 1], zeroline=False),
                                   boxmode='group')
                fig = go.Figure(data=data, layout=layout)
                fig.update_layout(title= "Normalized Current & Voltage<br><sup>Calculated under constant operating condition</sup>", title_x=0.5,title_y=0.90)
                fig.update_xaxes(title_text= "Month")
                fig['layout']['title']['font'] = dict(size=20)
                fig.update_layout(height=600)
                fig.update_layout(yaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                fig.update_layout(xaxis = dict(tickfont = dict(size=14), titlefont = dict(size = 16)))
                st.plotly_chart(fig, use_container_width=True, config = {'displaylogo': False})

                #st.write('Temperature coefficient of voltage (calculated) =', float(np.round(v_coef.tolist()[0],2)))
                #st.write('Temperature coefficient of voltage (reported) =', c_voltage)
            # elif end_date < start_date:
                # st.error('Star date is greater than the end date, please select correct dates')
            # else:
                # st.error('Data no available, please try with another start and end dates')
            # Download CSV file
            download_data = final_data[['datetime','Mod Temp(°C)','Current(A)','Voltage(V)','Power(W)']].copy()
            csv = download_data.to_csv()
            st.sidebar.download_button(label="Download CSV", data = csv,file_name='Device_Data.csv',mime='text/plain',)
            # with col2:
                # st.write('')
                # st.write('')
                # st.metric(label="Total Energy (kWh)/Panel", value= round(total_yield/1000, 2))
                # st.write('')
                # st.write('')
                # st.metric(label="Total Savings (INR)/Panel", value= round(total_yield*0.0085, 2))

elif authentication_status == False:
    st.error('Username/password is incorrect')
else:
    st.image('Screenshot (509).png')
    
    
    
############################################END of the Code################################################################################################

# Taking inputs

    # col1, col2, col3, col4, col5, col6 = st.columns((1.6, 1.6, 1.6, 0.9, 0.9, 1))
    # with col1:
        # st.image('SmartHelio logo (335 x 100 px)-2.png')
    # with col2:
        # # if name == "TATA Power":
            # # CCID = ['8991000905149874226f', '8991000905149874200f', '8991000904852270755f']
        # # elif name == "Ayana Power":
            # # CCID = ['8991102105471826640f', '8991000905280575269f', '8991000905280575277f']    
        # # elif name == "Mr. Shrivastava":
            # # CCID = ['8991000905606990044f']
        # sensor_ID = st.selectbox('Device ID', CCID)    
    # with col3:
        # option = st.selectbox('Select Analysis',('Energy Production','DC Power Time Series', 'Current Voltage Time Series', 'Temperature Coefficient'))
    # with col4:
        # start_date = st.date_input("Enter Start Date",(date.today() - timedelta(days=1)))
    # with col5:
        # end_date = st.date_input("Enter End Date",date.today())
        # end_date = end_date + timedelta(days = 1)
    # start_date = pd.to_datetime(start_date)
    # end_date = pd.to_datetime(end_date)

    # # Apply button which will run the main code

    # # if (col1.button("See Analysis")): #2
    # if sensor_ID not in list(sensor_data.sensor_ID):
        # st.error("Please Enter Correct Device ID")
    # else: 
       # # st.text(sensor_ID)
        
        # final_data1 = main(sensor_ID, start_date, end_date, sensor_data)
        # daily_yield, total_yield, output_energy = energy_yield(final_data1['Power(W)'])
        # final_data = pd.concat([final_data1,output_energy], axis = 1)
        # final_data.columns = ['Mod Temp(°C)','Current(A)','Voltage(V)','Power(W)', 'Energy Yield(kWh)']
        # # final_data['Hours'] = final_data.index.hour
        # # final_data['Hours'] = pd.DataFrame(final_data.strftime("%H:%M"))
        # final_data['Time(hhmm)'] = [datetime.strftime(d, "%H%M") for d in final_data.index]
        # # final_data['Time'] = [datetime.strftime(final_data.index,'%Y-%m-%dT%H:%M:%S')]
        # final_data['Time(hhmm)'] = final_data['Time(hhmm)'].astype(int)
        # # final_data['Time'] = pd.to_timedelta(final_data['Time'], format='%H:%M').iloc[0]
        # # final_data['Time'] = pd.to_datetime(final_data['Time'], format='%H:%M').iloc[0]
        # final_data['Date'] = final_data.index.date
        # #final_data['Minute'] = final_data.index.minute
        # #final_data['DateTime'] = str(final_data['Hours'])+":"+str(final_data['Minute'])
        # #final_data['DateTime'] = final_data.index.split("T")[1]
        # # st.write(final_data)
        # # Global_data = final_data
        
        # if len(final_data) != 0:
            # # st.success("Success")
            # # st.write(final_data)
            # with col6:
                # st.markdown('')
                # st.markdown('')
                # csv = final_data.to_csv()
                # st.download_button(label="Download CSV", data = csv,file_name='Device_Data.csv',mime='text/plain',) #2
                  
            # # Visualization of data
            # # Line chart of I,V and MT
            # # with col1:
            # # st.subheader('Plot Analysis')
            
            # final_data.reset_index(inplace = True)
            # # tab1, tab2, tab3, tab4, tab5, tab6 = st.columns((2, 1.5, 1, 1, 1, 1))
            # # with tab2:
                # # option = st.selectbox('',('Energy Production','DC Power Time Series', 'Current Voltage Time Series', 'Temperature Coefficient'))
            # plot1, plot2, plot3, plot4 = st.columns((5.5,0.3,1,1))
            # with plot1:
                # # option = st.selectbox('',('Energy Production','DC Power Time Series', 'Current Voltage Time Series', 'Temperature Coefficient'))
                # if timedelta(days=31) > (end_date - start_date) > timedelta(days=1) and option == 'Energy Production':
                   # dff = final_data[['datetime', 'Energy Yield(kWh)']].copy()
                   # dff.set_index('datetime', inplace = True)
                   # dff = dff.groupby(pd.Grouper(freq='d')).sum()
                   # fig = plt.figure(figsize=(10, 4))
                   # sns.barplot(x=dff.index, y=dff['Energy Yield(kWh)']/1000, color="#0A579E")
                   # xticks = [datetime.strftime(d, "%Y-%m-%d") for d in dff.index]
                   # plt.xticks(np.arange(0, len(xticks),1), xticks, rotation = 45)
                   # plt.title('Energy Production')
                   # plt.xlabel('Date', fontsize = 10)
                   # # energy_yield_new = final_data["Energy Yield(kWh)"].rolling(2).sum()
                   # # st.write(energy_yield_new)
                   # # sns.lineplot(x=energy_yield_new.index, y=energy_yield_new['Energy Yield(kWh)'], data = energy_yield_new.columns, color="#69b3a2")
                   # # plt.legend(labels=["Production"], loc='upper right', frameon=False)
                   # plt.locator_params(axis='x', nbins=18)
                   # #fig.autofmt_xdate()
                   # # sns.despine()
                   # st.pyplot(fig)
                   
                # elif (end_date - start_date) > timedelta(days=31) and option == 'Energy Production':
                   # dff = final_data[['datetime', 'Energy Yield(kWh)']].copy()
                   # dff.set_index('datetime', inplace = True)
                   # dff = dff.groupby(pd.Grouper(freq='W')).sum()
                   # fig = plt.figure(figsize=(10, 4))
                   # sns.barplot(x=dff.index, y=dff['Energy Yield(kWh)']/1000, color="#0A579E")
                   # xticks = [datetime.strftime(d, "%Y-%m-%d") for d in dff.index]
                   # plt.xticks(np.arange(0, len(xticks),1), xticks, rotation = 45)
                   # plt.title('Energy Production')
                   # plt.xlabel('Date', fontsize = 10)
                   # # energy_yield_new = final_data["Energy Yield(kWh)"].rolling(2).sum()
                   # # st.write(energy_yield_new)
                   # # sns.lineplot(x=energy_yield_new.index, y=energy_yield_new['Energy Yield(kWh)'], data = energy_yield_new.columns, color="#69b3a2")
                   # # plt.legend(labels=["Production"], loc='upper right', frameon=False)
                   # plt.locator_params(axis='x', nbins=18)
                   # #fig.autofmt_xdate()
                   # # sns.despine()
                   # st.pyplot(fig)
                   
                # elif (end_date - start_date) == timedelta(days=1) and option == 'Energy Production':
                   # dff = final_data[['datetime', 'Energy Yield(kWh)']].copy()
                   # dff.set_index('datetime', inplace = True)
                   # dff = dff.groupby(pd.Grouper(freq='H')).sum()
                   # fig = plt.figure(figsize=(10, 4))
                   # sns.barplot(x=dff.index, y=dff['Energy Yield(kWh)']/1000, color="#0A579E")
                   
                   # xticks = [datetime.strftime(d, "%H:%M") for d in dff.index]
                   # plt.xticks(np.arange(0, len(xticks),1), xticks, rotation = 45)
                   # plt.title('Energy Production')
                   # plt.xlabel('Time', fontsize = 10)
                   # # energy_yield_new = final_data["Energy Yield(kWh)"].rolling(2).sum()
                   # # st.write(energy_yield_new)
                   # # sns.lineplot(x=energy_yield_new.index, y=energy_yield_new['Energy Yield(kWh)'], data = energy_yield_new.columns, color="#69b3a2")
                   # # plt.legend(labels=["Production"], loc='upper right', frameon=False)
                   # plt.locator_params(axis='x', nbins=18)
                   # #fig.autofmt_xdate()
                   # # sns.despine()
                   # st.pyplot(fig)
                     
                # elif timedelta(days=31) > (end_date - start_date) > timedelta(days=1) and option == 'DC Power Time Series':
                   # dff = final_data[['datetime', 'Power(W)']].copy()
                   # dff.set_index('datetime', inplace = True)
                   # dff = dff.groupby(pd.Grouper(freq='d')).mean()
                   # fig = plt.figure(figsize=(10, 4))
                   # sns.barplot(x=dff.index, y=dff['Power(W)'], color="#0A579E")
                   # xticks = [datetime.strftime(d, "%Y-%m-%d") for d in dff.index]
                   # plt.xticks(np.arange(0, len(xticks),1), xticks, rotation = 45)
                   # plt.title('DC Average Power')
                   # plt.xlabel('Date', fontsize = 10)
                   # # energy_yield_new = final_data["Energy Yield(kWh)"].rolling(2).sum()
                   # # st.write(energy_yield_new)
                   # # sns.lineplot(x=energy_yield_new.index, y=energy_yield_new['Energy Yield(kWh)'], data = energy_yield_new.columns, color="#69b3a2")
                   # # plt.legend(labels=["Production"], loc='upper right', frameon=False)
                   # plt.locator_params(axis='x', nbins=18)
                   # #fig.autofmt_xdate()
                   # # sns.despine()
                   # st.pyplot(fig)
                   
                # elif (end_date - start_date) > timedelta(days=31) and option == 'DC Power Time Series':
                   # dff = final_data[['datetime', 'Power(W)']].copy()
                   # dff.set_index('datetime', inplace = True)
                   # dff = dff.groupby(pd.Grouper(freq='W')).mean()
                   # fig = plt.figure(figsize=(10, 4))
                   # sns.barplot(x=dff.index, y=dff['Power(W)'], color="#0A579E")
                   # xticks = [datetime.strftime(d, "%Y-%m-%d") for d in dff.index]
                   # plt.xticks(np.arange(0, len(xticks),1), xticks, rotation = 45)
                   # plt.title('DC Average Power')
                   # plt.xlabel('Date', fontsize = 10)
                   # # energy_yield_new = final_data["Energy Yield(kWh)"].rolling(2).sum()
                   # # st.write(energy_yield_new)
                   # # sns.lineplot(x=energy_yield_new.index, y=energy_yield_new['Energy Yield(kWh)'], data = energy_yield_new.columns, color="#69b3a2")
                   # # plt.legend(labels=["Production"], loc='upper right', frameon=False)
                   # plt.locator_params(axis='x', nbins=18)
                   # #fig.autofmt_xdate()
                   # # sns.despine()
                   # st.pyplot(fig)
                   
                # elif (end_date - start_date) == timedelta(days=1) and option == 'DC Power Time Series':
                   # # fig = plt.figure(figsize=(10, 4))
                   # # sns.lineplot(x=final_data['datetime'], y=final_data['Power(W)'], data = final_data.columns, color="#0A579E").set(title='DC Power')
                   # # # plt.legend(labels=["Power"], loc='upper right', frameon=False)
                   # # plt.xlabel('Date', fontsize = 10)
                   # # plt.locator_params(axis='x', nbins=18)
                   # # fig.autofmt_xdate()
                   # # # sns.despine()
                   # # st.pyplot(fig)
                   
                   # fig = final_data[['datetime', 'Power(W)']].set_index('datetime').plot(kind='line', figsize = (10,4))

                   # # dff = final_data[['datetime', 'Power(W)']].copy()
                   # # dff.set_index('datetime', inplace = True)
                   # # # dff = dff.groupby(pd.Grouper(freq='H')).sum()
                   # # fig = plt.figure(figsize=(10, 4))
                   # # sns.lineplot(x=dff.index, y=dff['Power(W)'], color="#0A579E")
                   
                   # # xticks = [datetime.strftime(d, "%H:%M") for d in dff.index]
                   # # plt.xticks(np.arange(0, len(xticks),1), xticks, rotation = 45)
                   # plt.title('DC Power Curve')
                   # plt.xlabel('Time', fontsize = 10)
                   # # energy_yield_new = final_data["Energy Yield(kWh)"].rolling(2).sum()
                   # # st.write(energy_yield_new)
                   # # sns.lineplot(x=energy_yield_new.index, y=energy_yield_new['Energy Yield(kWh)'], data = energy_yield_new.columns, color="#69b3a2")
                   # # plt.legend(labels=["Production"], loc='upper right', frameon=False)
                   # # plt.locator_params(axis='x', nbins=18)
                   # #fig.autofmt_xdate()
                   # # sns.despine()
                   # #st.pyplot(fig)
                   # # return fig
                   
                # elif timedelta(days=31) > (end_date - start_date) > timedelta(days=1) and option == 'Current Voltage Time Series':
                   # dff = final_data[['datetime', 'Current(A)', 'Voltage(V)']].copy()
                   # dff.set_index('datetime', inplace = True)
                   # dff = dff.groupby(pd.Grouper(freq='d')).mean()
                   # fig = plt.figure(figsize=(10, 4))
                   # # sns.barplot(x=dff.index, y=dff['Current(A)'], )
                   # sns.barplot(x=dff.index, y=dff['Voltage(V)'], color = "#0A579E")
                   # #plt.legend(labels=["Voltage" ], loc='upper left', frameon=False)
                   # sns.barplot(x=dff.index, y=dff['Current(A)'], color = 'orange',)
                   # xticks = [datetime.strftime(d, "%Y-%m-%d") for d in dff.index]
                   # plt.xticks(np.arange(0, len(xticks),1), xticks, rotation = 45)
                   # plt.title('Current Voltage Time Series')
                   # #plt.legend(labels=["Current","Voltage" ], loc='upper left', frameon=False)
                   # # plt.legend(labels=["Current"], loc='upper right', frameon=False)
                   # plt.xlabel('Date', fontsize = 10)
                   # plt.ylabel('Values', fontsize = 10)
                   # plt.ylim(0,42)
                   # # energy_yield_new = final_data["Energy Yield(kWh)"].rolling(2).sum()
                   # # st.write(energy_yield_new)
                   # # sns.lineplot(x=energy_yield_new.index, y=energy_yield_new['Energy Yield(kWh)'], data = energy_yield_new.columns, color="#69b3a2")
                   # # plt.legend(labels=["Production"], loc='upper right', frameon=False)
                   # #plt.locator_params(axis='x', nbins=18)
                   # #fig.autofmt_xdate()
                   # # sns.despine()
                   # st.pyplot(fig)
                   
                   # # st.text('')
                   # # final_data.reset_index(inplace = True)
                   # # fig = plt.figure(figsize=(10, 4))
                   # # sns.lineplot(x=final_data['Voltage(V)'], y=final_data['Current(A)'], color = 'steelblue')
                   # # plt.title('PV IV curve', weight = 'bold')
                   # # plt.xlim(0,42)
                   
                   # # st.pyplot(fig)
                   # # st.text('')
                   # # #final_data.reset_index(inplace = True)
                   # # fig = plt.figure(figsize=(10, 4))
                   # # sns.lineplot(x=np.arange(0, len(final_data['Power(W)']),1), y=final_data['Power(W)'], color = 'steelblue')
                   # # plt.title('PV Power curve', weight = 'bold')

                   
                   # # st.pyplot(fig)
                   
                # elif (end_date - start_date) > timedelta(days=31) and option == 'Current Voltage Time Series':
                   # dff = final_data[['datetime', 'Current(A)', 'Voltage(V)']].copy()
                   # dff.set_index('datetime', inplace = True)
                   # dff = dff.groupby(pd.Grouper(freq='W')).mean()
                   # fig = plt.figure(figsize=(10, 4))
                   # # sns.barplot(x=dff.index, y=dff['Current(A)'], )
                   # sns.barplot(x=dff.index, y=dff['Voltage(V)'], color = "#0A579E")
                   
                   # #plt.legend(labels=["Voltage" ], loc='upper left', frameon=False)
                   # sns.barplot(x=dff.index, y=dff['Current(A)'], color = 'orange',)
                   # xticks = [datetime.strftime(d, "%Y-%m-%d") for d in dff.index]
                   # plt.xticks(np.arange(0, len(xticks),1), xticks, rotation = 45)
                   # plt.title('Current Voltage Time Series')
                   # #plt.legend(labels=["Current","Voltage" ], loc='upper left', frameon=False)
                   # # plt.legend(labels=["Current"], loc='upper right', frameon=False)
                   # plt.xlabel('Date', fontsize = 10)
                   # plt.ylabel('Values', fontsize = 10)
                   # plt.ylim(0,42)
                   # st.pyplot(fig)
                # elif (end_date - start_date) == timedelta(days=1) and option == 'Current Voltage Time Series':
                    # fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
                    # sns.lineplot(x=final_data['Time(hhmm)'], y=final_data['Voltage(V)'], ax = ax1, color="#0A579E")
                    # sns.lineplot(x=final_data['Time(hhmm)'], y=final_data['Current(A)'], ax = ax2, color="#0A579E")
                    # fig.suptitle('IV Time Series Plot')
                    # st.pyplot(fig) 
                    
                # elif timedelta(days=31) > (end_date - start_date) > timedelta(days=1) and option == 'Temperature Coefficient':
                    # fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
                    # sns.scatterplot(x=final_data['Mod Temp(°C)'], y=final_data['Voltage(V)'],c=final_data['Voltage(V)'],  s=7, ax = ax1)
                    # sns.scatterplot(x=final_data['Mod Temp(°C)'], y=final_data['Current(A)'],c=final_data['Current(A)'],  s=7, ax = ax2)
                    # fig.suptitle('Temperature Analysis')
                    # st.pyplot(fig)
                    # # # fig, (ax1, ax2) = plt.subplots(figsize=(50,100))
                    # # # fig.figure(figsize=(10, 4))
                    # # #fig.set_aspect(1.5)
                    # # fig.suptitle('Horizontally stacked subplots')
                    # # ax1.scatter(final_data['Mod Temp(°C)'], final_data['Current(A)'])
                    # # ax2.scatter(final_data['Mod Temp(°C)'], final_data['Voltage(V)'])
                    # # st.pyplot(fig)
                    
                    # # fig = plt.figure()
                    # # ax1 = fig.add_subplot(1,1,1, adjustable='box', aspect=0.5)
                    # # ax2 = fig.add_subplot(1,2,1, adjustable='box', aspect=0.5)

                    # # ax1.scatter(final_data['Mod Temp(°C)'], final_data['Current(A)'])
                    # # ax2.scatter(final_data['Mod Temp(°C)'], final_data['Voltage(V)'])

                    # # st.pyplot(fig)
                    # # fig = plt.figure(figsize = (4,4))
                  # # # final_data.columns = ['datetime','Mod Temp(°C)','Current(A)','Voltage(V)','Power(W)']
                    # # fig = sns.JointGrid(
                    # # x=final_data['Mod Temp(°C)'], y=final_data['Voltage(V)'])
                    # # fig.plot_joint(plt.scatter, c=final_data['Voltage(V)'], label='check', cmap="Blues", s=7)
                    # # fig.plot_marginals(sns.histplot, kde=True, color="steelblue")
                    # #st.pyplot(fig)
                # elif (end_date - start_date) > timedelta(days=31) and option == 'Temperature Coefficient':
                    # fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
                    # sns.scatterplot(x=final_data['Mod Temp(°C)'], y=final_data['Voltage(V)'],c=final_data['Voltage(V)'],  s=7, ax = ax1)
                    # sns.scatterplot(x=final_data['Mod Temp(°C)'], y=final_data['Current(A)'],c=final_data['Current(A)'],  s=7, ax = ax2)
                    # fig.suptitle('Temperature Analysis')
                    # st.pyplot(fig)
                # elif (end_date - start_date) == timedelta(days=1) and option == 'Temperature Coefficient':
                    # fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
                    # sns.scatterplot(x=final_data['Mod Temp(°C)'], y=final_data['Voltage(V)'],c=final_data['Voltage(V)'],  s=7, ax = ax1)
                    # sns.scatterplot(x=final_data['Mod Temp(°C)'], y=final_data['Current(A)'],c=final_data['Current(A)'],  s=7, ax = ax2)
                    # fig.suptitle('Temperature Analysis')
                    # st.pyplot(fig)
            # with plot3:
            
                
                # st.text('')
                # st.text('')
                # st.metric(label="Total Energy(kWh)", value= round(total_yield/1000, 2))
            # with plot4:
                
                # st.text('')
                # st.text('')
                # # st.markdown(f'<p style="background-color:#ffffff;color:#33ff33;font-size:24px;border-radius:2%;">{"Hwllo"}</p>', unsafe_allow_html=True)
                # st.metric(label="Total Savings(INR)", value= round(total_yield*0.0065, 2))
            
            
                    
                    
                

                

                     
                # # # final_data.columns = ['datetime','Mod Temp(°C)','Current(A)','Voltage(V)','Power(W)', 'Energy Yield(kWh)']
                # # # final_data['']
                # # # date_d,time_t = final_data['datetime'].split()
                # # with plot1:
                   # # if (end_date - start_date) > timedelta(days=1):
                       # # #st.markdown('Energy Production')
                       # # dff = final_data[['datetime', 'Energy Yield(kWh)']].copy()
                       # # dff.set_index('datetime', inplace = True)
                       # # dff = dff.groupby(pd.Grouper(freq='W')).sum()
                       # # fig = plt.figure(figsize=(10, 5))
                       # # sns.barplot(x=dff.index, y=dff['Energy Yield(kWh)'], color="steelblue")
                       # # xticks = [datetime.strftime(d, "%Y-%m-%d") for d in dff.index]
                       # # plt.xticks(np.arange(0, len(xticks),1), xticks, rotation = 45)
                       # # plt.title('Energy Production', weight = 'bold')
                       # # plt.xlabel('Date', fontsize = 10)
                       # # # energy_yield_new = final_data["Energy Yield(kWh)"].rolling(2).sum()
                       # # # st.write(energy_yield_new)
                       # # # sns.lineplot(x=energy_yield_new.index, y=energy_yield_new['Energy Yield(kWh)'], data = energy_yield_new.columns, color="#69b3a2")
                       # # # plt.legend(labels=["Production"], loc='upper right', frameon=False)
                       # # plt.locator_params(axis='x', nbins=18)
                       # # #fig.autofmt_xdate()
                       # # sns.despine()
                       # # st.pyplot(fig)
                       
                       
                       # # #st.markdown('DC Power')
                       # # fig = plt.figure(figsize=(10, 5))
                       
                       # # sns.lineplot(x=final_data['Date'], y=final_data['Power(W)'], data = final_data.columns, color="#69b3a2").set(title='DC Power')
                       # # # sns.barplot(x='Date', y='Power(W)'], data = final_data.columns, color="#69b3a2")
                       # # plt.legend(labels=["Power"], loc='upper right', frameon=False)
                       # # fig.autofmt_xdate()
                       # # sns.despine()
                       # # st.pyplot(fig)
                       
                       
                       # # #st.markdown('IV Curve')
                       # # fig = plt.figure(figsize=(10, 5))
                       # # # sns.lineplot(x=final_data['Date'], y=final_data['Current(A)'], data = final_data.columns, color="#69b3a2")
                       # # sns.lineplot(x=final_data['Date'], y=final_data['Current(A)'], data = final_data.columns, color="#69b3a2").set(title='VI Curve')
                       # # plt.legend(labels=["Current","Freq"], loc='upper left', frameon=False)
                       # # # sns.lineplot(data=df.column1, color="g")
                       # # ax2 = plt.twinx()
                       # # sns.lineplot(x=final_data['Date'], y=final_data['Voltage(V)'], data = final_data.columns, color="olive", ax=ax2)
                       # # # sns.lineplot(x=final_data['Date'], y=final_data['Voltage(V)'], data = final_data.columns, color="olive")
                       # # plt.legend(labels=["Voltage", "Freq"], loc='upper right', frameon=False)
                       # # fig.autofmt_xdate()
                       # # # sns.despine()
                       # # st.pyplot(fig)
                       # # # final_data.reset_index("datetime",inplace = True)
                       # # # st.markdown("Daily Energy Yield")
                       # # final_data.set_index("datetime",inplace = True)
                       # # fig = plt.figure(figsize=(10, 5))
                       # # daily_yield,_,_ = energy_yield(final_data['Power(W)'])
                       # # # st.write(daily_yield)
                       # # # st.markdown('Energy Yield')
                       # # # daily_yield.column = ['Energy(kWh)']
                       # # # st.write(daily_yield)
                       
                       # # # daily_yield['Date'] = daily_yield.index.date
                       # # # daily_yield = pd.DataFrame(daily_yield)
                       # # # # daily_yield = daily_yield.astype({"Energy(kWh)": float})
                       # # # daily_yield.reset_index(inplace = True)
                       # # # daily_yield.columns = ['Date','Energy(kWh)']
                       # # # st.write(daily_yield)
                       # # # # daily_yield = daily_yield.astype({"Energy(kWh)": float})
                       # # # # sns.barplot(x=daily_yield['Date'], y=daily_yield['Energy(kWh)'], data=daily_yield.columns)
                       # # # sns.barplot(x ='Date', y ='Energy(kWh)', data=daily_yield)
                       # # # fig.autofmt_xdate()
                       # # # sns.despine()
                       # # # st.pyplot(fig)       
                   # # elif (end_date - start_date) == timedelta(days=1):
                       # # #day_data = final_data[final_data['Date'] == end_date]
                       # # # st.markdown('Energy Production')
                       # # fig = plt.figure(figsize=(10, 5))
                       # # sns.lineplot(x=final_data['Time(hhmm)'], y=final_data['Energy Yield(kWh)'], data = final_data.columns, color="#69b3a2").set(title='Energy Production')
                       # # # energy_yield_new = final_data["Energy Yield(kWh)"].rolling(2).sum()
                       # # # st.write(energy_yield_new)
                       # # # sns.lineplot(x=energy_yield_new.index, y=energy_yield_new['Energy Yield(kWh)'], data = energy_yield_new.columns, color="#69b3a2")
                       # # plt.legend(labels=["Production"], loc='upper right', frameon=False)
                       # # plt.locator_params(axis='x', nbins=18)
                       # # # fig.autofmt_xdate()
                       # # sns.despine()
                       # # st.pyplot(fig)
                       
                       
                       # # # st.markdown('DC Power')
                       # # fig = plt.figure(figsize=(10, 5))
                       # # sns.lineplot(x=final_data['Time(hhmm)'], y=final_data['Power(W)'], data = final_data.columns, color="#69b3a2").set(title='DC Power')
                       # # plt.legend(labels=["Power"], loc='upper right', frameon=False)
                       # # plt.locator_params(axis='x', nbins=18)
                       # # # fig.autofmt_xdate()
                       # # sns.despine()
                       # # st.pyplot(fig)
                       
                       # # # st.markdown('IV Curve')
                       # # fig = plt.figure(figsize=(10, 5))
                       # # sns.lineplot(x=final_data['Time(hhmm)'], y=final_data['Current(A)'], data = final_data.columns, color="#69b3a2").set(title='VI Curve')
                       # # plt.legend(labels=["Current"], loc='upper left', frameon=False)
                       # # # sns.lineplot(data=df.column1, color="g")
                       # # ax2 = plt.twinx()
                       # # sns.lineplot(x=final_data['Time(hhmm)'], y=final_data['Voltage(V)'], data = final_data.columns, color="olive", ax=ax2)
                       # # # sns.lineplot(x=final_data['Time(hhmm)'], y=final_data['Voltage(V)'], data = final_data.columns, color="olive")
                       # # plt.legend(labels=["Voltage"], loc='upper right', frameon=False)
                      
                       # # # sns.despine()
                       # # plt.locator_params(axis='x', nbins=18)
                       # # st.pyplot(fig)
                  # # # st.markdown('Current and Voltage')
                 # # #  st.line_chart(data = final_data.iloc[:,:-1])
                  # # # daily_yield,output_energy,_ = energy_yield(final_data['Power(W)'])
                   
                 # # #  st.markdown('DC Power')
                 # # #  st.bar_chart(final_data['Power(W)'])
                # # with plot2:
                   # # final_data.reset_index(inplace = True) 
                   # # st.markdown('     Temperature Vs Voltage')
                   # # fig = plt.figure(figsize = (8,5))
                  # # # final_data.columns = ['datetime','Mod Temp(°C)','Current(A)','Voltage(V)','Power(W)']
                   # # fig = sns.JointGrid(
                   # # x=final_data['Mod Temp(°C)'], y=final_data['Voltage(V)'])
                   # # fig.plot_joint(
                   # # plt.scatter, c=final_data['Power(W)'], label='check', cmap="Blues", s=7)
                   # # fig.plot_marginals(sns.histplot, kde=True, color="steelblue")
                   # # st.pyplot(fig)
                    
                   # # st.markdown('    Temperature Vs Current')
                   # # fig = sns.JointGrid(
                   # # x=final_data['Mod Temp(°C)'], y=final_data['Current(A)'])
                   # # fig.plot_joint(
                   # # plt.scatter, c=final_data['Power(W)'], label='check', cmap="Blues", s=7)
                   # # fig.plot_marginals(sns.histplot, kde=True, color="steelblue")
                   # # st.pyplot(fig)
                
                   # # # sns.lineplot(data=df.column1, color="g")
                   # # # ax2 = plt.twinx()
                   # # # sns.lineplot(data=df.column2, color="b", ax=ax2)
                    # # # fill="#69b3a2", color="#e9ecef", alpha=0.9
                            
                   # # # # st.markdown('IV Curve')
                   # # # #fig = plt.figure(figsize=(10, 4))
                   # # # sns.lineplot(x=final_data['datetime'], y=final_data['Current(A)'], data = final_data.columns, color="#69b3a2")
                   # # # plt.legend(labels=["Current"], loc='upper left', frameon=False)
                   # # # # sns.lineplot(data=df.column1, color="g")
                   # # # ax2 = plt.twinx()
                   # # # sns.lineplot(x=final_data['datetime'], y=final_data['Voltage(V)'], data = final_data.columns, color="olive", ax=ax2)
                   # # # plt.legend(labels=["Voltage"], loc='upper right', frameon=False)
                   # # # st.pyplot(fig)
                
               
                
          
                # # # st.line_chart(data = final_data.iloc[:,:-1])
                # # # st.line_chart(data = final_data.iloc[:,:-1])
                # # # bar chart of power
                # # # st.subheader('Daily Power')
                # # # st.bar_chart(final_data['Power(W)'])
        # # #             if (st.button("GENERATE REPORT")):
        # # #                 st.success('Your report will be ready for download in a moment!')  

            # # else:
                # # st.error("Error : No data found in the given date range. Please try another date range!")
                # # # st.error("")
                    
    # # # plot1, plot2 = st.columns(2)
    # # # with plot1:
    # # #     st.line_chart(data = Global_data.iloc[:,:-1])
    # # # with plot2:
    # #     st.bar_chart(Global_data['Power(W)'])




# lines = []
# with open('user_data.txt') as f:
    # lines = f.readlines()

# count = 0
# for line in lines:
    # count += 1
    # st.write(f'line {count}: {line}')   
    

