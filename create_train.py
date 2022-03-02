"""
雨のnetCDF : rain_nc/
土地利用のnetCDF : geo_nc/
の2種の2次元数値データ群と
水位,流域面積,川幅などの1次元数値データ群から
機械学習のための入出力データセットを作成する.

location(str)を渡すことで,
対応するnetCDFを読みにいき,
durings(list)で作成するデータ期間を指定する.

想定している入力データは以下の通り.
input--
      |-  rain  : array(m,n,18) : m*nのグリッド雨量を18タイムステップ(3時間)
      |-  land  : array(w,h)    : w*hのカテゴリカルグリッドデータ
      |-  water : array(3)          : 現在までの3時間分の水位
      |-  area  : float             : 流域面積(km^2)
      |-  width : float             : 川幅(m)

output--
      |- water_  : float             : 3時間後の水位変化

関数一覧
create_data--
      |
      |--create_short_data()
      |            |-get_rain_short_dat()
      |            |-get_river_short_dat()
      |      
      |--create_long_data()
                   |-get_rain_long_dat()
                   |-get_river_short_dat()
            
"""

import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
import joblib
from PIL import Image
from tqdm import tqdm
import json

#get_rain_short_datの時にcumulativeを渡す
#start = target-dt.timedelta(hours = 9 + cumulative)

def get_rain_short_dat(rain_path,location,target,period,cumulative=[]):
      #target を　UTC に変換してから
      target = target - dt.timedelta(hours=9)
      #start:予測の開始ではなく, 累積の開始
      if len(cumulative)!=0:
            start = np.datetime64(target - dt.timedelta(days=1,hours=max(cumulative)))
      else:
            start = np.datetime64(target - dt.timedelta(days=1,hours=5))
      end = np.datetime64(target + period)
      
      if target.year != (target + period).year:
            temp_end = dt.datetime(year=target.year,month=12,day=31,hour=23,minute=50)
            temp_start = temp_end + dt.timedelta(minutes=10)
            with xr.open_dataset(rain_path + str(target.year) + "/" + str(target.month) + "/" + str(location) + ".nc") as rain_nc1:
                  temp_nc1 = rain_nc1.sel(time=slice(start,np.datetime64(temp_end)))
            with xr.open_dataset(rain_path + str(target.year+1) + "/" + str(1) + "/" + str(location) + ".nc") as rain_nc2:
                  temp_nc2 = rain_nc2.sel(time=slice(np.datetime64(temp_start),end))
            dur_rain_nc = xr.concat([temp_nc1,temp_nc2],dim="time")

      elif target.month != (target + period).month:
            temp_start = dt.datetime(year=target.year,month=target.month+1,day=1,hour=0,minute=0)
            temp_end = temp_start - dt.timedelta(minutes=10)
            with xr.open_dataset(rain_path + str(target.year) + "/" + str(target.month) + "/" + str(location) + ".nc") as rain_nc1:
                  temp_nc1 = rain_nc1.sel(time=slice(start,np.datetime64(temp_end)))
            with xr.open_dataset(rain_path + str(target.year) + "/" + str(target.month+1) + "/" + str(location) + ".nc") as rain_nc2:
                  temp_nc2 = rain_nc2.sel(time=slice(np.datetime64(temp_start),end))
            dur_rain_nc = xr.concat([temp_nc1,temp_nc2],dim="time")
      else:
            with xr.open_dataset(rain_path + str(target.year) + "/" + str(target.month) + "/" + str(location) + ".nc") as rain_nc:
                  dur_rain_nc = rain_nc.sel(time=slice(start,end))
            
      return dur_rain_nc

def get_river_short_dat(river_path,location,target,period):
      start = np.datetime64(target-dt.timedelta(days=1))
      end = np.datetime64(target + period)

      if target.year != (target + period).year:
            temp_end = dt.datetime(year=target.year,month=12,day=31,hour=23)
            temp_start = temp_end + dt.timedelta(hours=1)
            riv_df1 = pd.read_csv(river_path + str(target.year) + "/" + str(target.month) + "/" + str(location) + ".csv",index_col=0)
            riv_df2 = pd.read_csv(river_path + str(target.year+1) + "/" + str(1) + "/" + str(location) + ".csv",index_col=0)
            riv_df1.index = pd.to_datetime(riv_df1.index)
            riv_df2.index = pd.to_datetime(riv_df2.index)
            dur_river_df = pd.concat([riv_df1.loc[start:temp_end],riv_df2.loc[temp_start:end]])            

      elif target.month != (target + period).month:
            temp_start = dt.datetime(year=target.year,month=target.month+1,day=1,hour=0,minute=0)
            temp_end = temp_start - dt.timedelta(minutes=10)
            riv_df1 = pd.read_csv(river_path + str(target.year) + "/" + str(target.month) + "/" + str(location) + ".csv",index_col=0)
            riv_df2 = pd.read_csv(river_path + str(target.year) + "/" + str(target.month+1) + "/" + str(location) + ".csv",index_col=0)
            riv_df1.index = pd.to_datetime(riv_df1.index)
            riv_df2.index = pd.to_datetime(riv_df2.index)
            dur_river_df = pd.concat([riv_df1.loc[start:temp_end],riv_df2.loc[temp_start:end]])
      else:
            riv_df = pd.read_csv(river_path + str(target.year) + "/" + str(target.month) + "/" + str(location) + ".csv",index_col=0)
            riv_df.index = pd.to_datetime(riv_df.index)
            dur_river_df = riv_df.loc[start:end]
      return dur_river_df

def get_rain_long_dat(rain_path,location,durings_long):
      count=0
      rain_nc = xr.Dataset()
      for year in range(durings_long[0].year,durings_long[1].year+1):
            if year != durings_long[1].year:
                  for month in range(1,12):
                        with xr.open_dataset(rain_path + str(year) + "/" + str(month) + "/" + str(location) + ".nc") as temp_rain_nc:
                              if count == 0:
                                    rain_nc = temp_rain_nc
                                    count=1
                              else :
                                    rain_nc = xr.concat([rain_nc,temp_rain_nc],dim="time")
            else:
                  for month in range(1,durings_long[1].month+1):
                        with xr.open_dataset(rain_path + str(year) + "/" + str(month) + "/" + str(location) + ".nc") as temp_rain_nc: 
                              if count == 0:
                                    rain_nc = temp_rain_nc
                                    count=1
                              else :
                                    rain_nc = xr.concat([rain_nc,temp_rain_nc],dim="time")
      return rain_nc.sel(time=slice(durings_long[0],durings_long[1]))

def get_river_long_dat(river_path,location,durings_long):
      count=0
      riv_df = pd.DataFrame()
      for year in range(durings_long[0].year,durings_long[1].year+1):
            if year != durings_long[1].year:
                  for month in range(1,13):
                        temp_riv_df = pd.read_csv(river_path + str(year) + "/" + str(month) + "/" + str(location) + ".csv",index_col=0)
                        if count == 0:
                              riv_df = temp_riv_df
                              count=1
                        else :
                              riv_df = pd.concat([riv_df.loc[:],temp_riv_df.loc[:]])
            else:
                  for month in range(1,durings_long[1].month+1):
                        temp_riv_df = pd.read_csv(river_path + str(year) + "/" + str(month) + "/" + str(location) + ".csv",index_col=0)
                        if count == 0:
                              riv_df = temp_riv_df
                              count=1
                        else :
                              riv_df = pd.concat([riv_df.loc[:],temp_riv_df.loc[:]])
      riv_df.index = pd.to_datetime(riv_df.index)
      return riv_df.loc[durings_long[0]:durings_long[1]]

def create_data(dur_rain_nc,dur_river_df,x_rain,x_river,x_geo,y,geo_nc,lead_time,window,cumulative=[]):
      temp_y = (dur_river_df.astype(float) - dur_river_df.shift(lead_time).astype(float)).shift(-lead_time).dropna()
      #temp_y =(dur_river_df.shift(-lead_time).astype(float)).dropna()
      """
      try:
            temp_y = (dur_river_df - dur_river_df.shift(lead_time)).dropna()
      except:
            pd.set_option('display.max_rows', None)
            print(dur_river_df.shift(lead_time).dropna().dtypes)
      """
      temp_y["time"] = pd.to_datetime(temp_y.index)
      temp_y.set_index("time",inplace=True)
      start_prediction = temp_y.index[0]
      end_prediction = temp_y.index[len(temp_y)-1]
      #nowは現在時刻, ただしtemp_yは既に([now+lt]-[now])がnowに割当(将来の水位変化)
      for now in tqdm(dur_river_df.index[12:-12-lead_time]):
            cumulative_rain=[]
            # start end は UTCに変えて(-9),start:end(now)で現在までの雨量
            start = np.datetime64(now-dt.timedelta(hours=9+window-1,minutes=50))
            #start = np.datetime64(now-dt.timedelta(hours=9+window))
            end = np.datetime64(now-dt.timedelta(hours=9))

            temp_rain_nc = dur_rain_nc.rain.sel(time=slice(start,end))
            sh = len(dur_rain_nc.latitude)*len(dur_rain_nc.longitude)
            #if 累積雨量をデータに加えるなら
            if len(cumulative)!=0:
                  for cum in cumulative:
                        start_cumulative = np.datetime64(now-dt.timedelta(hours=9+cum-1,minutes=50))
                        cumulative_nc = dur_rain_nc.rain.sel(time=slice(start_cumulative,end))
                        cumulative_rain.append([np.sum(cumulative_nc.values)/(6*sh)])
                  """
                  for j in range(window):
                        for cum in cumulative:
                              start_cumulative = np.datetime64(now-dt.timedelta(hours=9+cum-1+j,minutes=50))
                              end_cumulative = np.datetime64(now-dt.timedelta(hours=j))
                              cumulative_nc = dur_rain_nc.rain.sel(time=slice(start_cumulative,end_cumulative))
                              cumulative_rain.append([np.sum(cumulative_nc.values)/(6*sh*255*cum)])
                              #print(start_cumulative)
                              #print(end_cumulative)
                              #print(now)
                              #print("--------------------")
                  """
            # 現在までの水位
            temp_river_df = dur_river_df[now-dt.timedelta(hours=window-1):now]
            temp_geo = get_target_geo(geo_nc,now)
            if dt.datetime(2012,6,29) <now < dt.datetime(2012,7,1):
                  continue
            """
            if len(temp_rain_nc) !=18 :
                  nan_list = find_nan(temp_rain_nc,now-dt.timedelta(hours=3))
                  temp_rain_nc = fill_rain(dur_rain_nc,temp_rain_nc,nan_list)
                  #まず,穴あきの場所を示した配列を用意(nan_list)
                  #for文を用いて前後の欠損していない値をそれぞれ探し出す.
                  #欠損値から何個離れていたか->前:m 後:nとして, 重み付けを行なって補間
                  x_rain.append(temp_rain_nc)
            else :
                  x_rain.append(temp_rain_nc.values)
            """
            temp_x_1d = np.array(temp_river_df.values)
            for p in cumulative_rain:
                  temp_x_1d = np.append(temp_x_1d,p)
            
            if len(temp_rain_nc) == window*6 and len(temp_river_df.values) == window and now >= start_prediction and now <= end_prediction:
                  x_geo.append(temp_geo)
                  x_rain.append(temp_rain_nc.values)
                  x_river.append(temp_x_1d)
                  #x_river.append(temp_river_df.values)
                  y.append(temp_y[now:now].values[0])
      return x_rain,x_river,x_geo,y
      
def create_short_data(root_path,location,durings,period,geo_nc,lead_time,window):
      x_rain, x_river, x_geo, y = [],[],[],[]
      tmp_x_rain, tmp_x_river, tmp_y = [],[],[]
      rain_path = root_path + "rain_nc/"
      river_path = root_path + "river_csv/"
      for target in durings:
            dur_river_df = get_river_short_dat(river_path,location,target,period)
            dur_rain_nc = get_rain_short_dat(rain_path,location,target,period)
            #print(dur_rain_nc.time)
            x_rain, x_river, x_geo,y = create_data(dur_rain_nc,dur_river_df,x_rain,x_river,x_geo,y,geo_nc,lead_time,window)            
      return x_rain,x_river,x_geo,y

def create_short_data2(root_path,location,durings,period,geo_nc,lead_time,window,cumulative=[]):
      x_rain, x_river, x_geo, y = [],[],[],[]
      tmp_x_rain, tmp_x_river, tmp_y = [],[],[]
      rain_path = root_path + "rain_nc/"
      river_path = root_path + "river_csv/"
      for target in durings:   
            dur_river_df = get_river_short_dat(river_path,location,target,period)
            dur_rain_nc = get_rain_short_dat(rain_path,location,target,period,cumulative)
            x_rain, x_river, x_geo,y = create_data(dur_rain_nc,dur_river_df,x_rain,x_river,x_geo,y,geo_nc,lead_time,window,cumulative)            
      return x_rain,x_river,x_geo,y

def create_long_data(root_path,location,during,geo_nc,lead_time,window):
      x_rain, x_river,x_geo, y = [],[],[],[]
      rain_path = root_path + "rain_nc/"
      river_path = root_path + "river_csv/"

      dur_rain_nc = get_rain_long_dat(rain_path,location,during)
      dur_river_df = get_river_long_dat(river_path,location,during)

      x_rain,x_river,x_geo,y = create_data(dur_rain_nc,dur_river_df,x_rain,x_river,x_geo,y,geo_nc,lead_time,window)      
      return x_rain,x_river,x_geo,y

def subtract_list(lst1, lst2):
    lst = lst1.copy()
    for element in lst2:
        try:
            lst.remove(element)
        except ValueError:
            pass
    return lst

def find_nan(rain_nc,start):
      correct=[]
      for diff in range(10,190,10):
            correct.append(np.datetime64(start+dt.timedelta(minutes=diff)))
      nan_list=subtract_list(correct,rain_nc["time"].values)
      return nan_list

def fill_rain(root_rain,temp_rain,nan_list):
      rains=temp_rain.values.tolist()
      for days in reversed(nan_list):
            if immpossible == 1:
                  impossible=0
                  continue
            days = pd.to_datetime(days)
            n=1
            m=1
            while((days+dt.timedelta(minutes=n*10)) not in root_rain["time"].values):
                  n+=1
                  if n>20:
                        impossible=1
                        break
            while((days-dt.timedelta(minutes=m*10)) not in root_rain["time"].values):
                  m+=1
                  if m>20:
                        immpossible=1
                        break
            forward_rain = root_rain.rain.sel(time=np.datetime64(days+dt.timedelta(minutes=n*10))).values
            backward_rain = root_rain.rain.sel(time=np.datetime64(days-dt.timedelta(minutes=m*10))).values
            lack_rain = (n*forward_rain + m*backward_rain)/(n+m)
            position = 0
            add_count=0
            for target_time in temp_rain["time"].values:
                  if pd.to_datetime(target_time) > days:
                        rains.insert(position,lack_rain.tolist())
                        add_count=1
                        break
                  else :
                        position += 1
            if add_count==0:      
                  rains.insert(position,lack_rain.tolist())
      return np.array(rains)

def reshape_geo(geo):
      img = Image.fromarray(np.array(geo))
      img = img.resize((64,64))
      new_geos= np.array(img)
      return new_geos

def get_target_geo(nc,day):
      if day.year <= 2011:
            target_geo = nc["ver1609"].values
      elif day.year <= 2016:
            target_geo = nc["ver1803"].values
      else:
            target_geo = nc["ver2103"].values
      geo = reshape_geo(target_geo)
      return geo

def str_to_datetime(durings):
      new_durings = []
      for temp_str in durings:
            temp_datetime = dt.datetime.strptime(temp_str,'%Y/%m/%d')
            new_durings.append(temp_datetime)
      return new_durings

if __name__ == "__main__":
      root_path = "/home/student/e17/e175764/"
      rain_path = root_path + "rain_nc/"
      river_path = root_path + "river_csv/"
      geo_path = root_path + "geo_nc/"
      lead_time = 3
      period = dt.timedelta(days=3,hours=lead_time) #開始日から何日分のデータでやるのか
      location = "ishida"  #1地点毎に作って,あとで結合する
      window = 5#何時間分の雨を考慮するか 
      info_json = json.load(open(root_path + "info.json"))
      temp_durings = info_json[location]["train_during"]
      durings = str_to_datetime(temp_durings)
      #during = [dt.datetime(2010,1,1),dt.datetime(2013,7,31,3)]      
      
      geo_nc = xr.open_dataset(geo_path + location + ".nc")
      x_rain, x_river, x_geo, y = [],[],[],[]
      #x_rain,x_river,y = create_long_data(root_path,location,during)
      #x_rain,x_river,x_geo,y = create_short_data(root_path,location,durings,period,geo_nc,lead_time,window)
      x_rain,x_river,x_geo,y = create_short_data2(root_path,location,durings,period,geo_nc,lead_time,window,[8,9,10])

      """
      geo = geo_nc["ver1803"].values
      geo = reshape_geo(geo)
      x_geo = [geo for i in range(len(y))]
      """
      
      with open(root_path + "train/x_rain/" + location+".jb",mode="wb") as f:
            joblib.dump(x_rain,f,compress=3)
      with open(root_path + "train/x_river/" + location+".jb",mode="wb") as f:
            joblib.dump(x_river,f,compress=3)
      with open(root_path + "train/x_geo/" + location +".jb",mode="wb") as f:
            joblib.dump(x_geo,f,compress=3)
      with open(root_path + "train/y_data/" + location+".jb",mode="wb") as f:
            joblib.dump(y,f,compress=3)

      geo_nc.close()
