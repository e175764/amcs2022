import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
import joblib
from PIL import Image
from create_train import create_short_data,create_short_data2,reshape_geo
import json

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
      lt=3
      period = dt.timedelta(days=3,hours=lt) #開始日から何日分のデータでやるのか
      location = "ishida"  #1地点毎に作って,あとで結合する
      window = 5#何時間分の雨を考慮するか
      info_json = json.load(open(root_path + "info.json"))
      temp_durings = info_json[location]["test_during"]
      durings = str_to_datetime(temp_durings)
      """
      durings = [   #期間の開始日
                  dt.datetime(2018,7,4)
                  #dt.datetime(2017,8,13),
                  #dt.datetime(2017,11,12)
                  ]
      """
      geo_nc = xr.open_dataset(geo_path + location + ".nc")
      x_rain, x_river, x_geo, y = [],[],[],[]
      #x_rain,x_river,y = create_long_data(root_path,location,during)
      #x_rain,x_river,x_geo,y = create_short_data(root_path,location,durings,period,geo_nc,lt,window)
      x_rain,x_river,x_geo,y = create_short_data2(root_path,location,durings,period,geo_nc,lt,window,[8,9,10])
      """
      geo = geo_nc["ver1803"].values
      geo = reshape_geo(geo)
      x_geo = [geo for i in range(len(y))]
      """
      with open(root_path + "test/"+location+"/x_rain/" + location+".jb",mode="wb") as f:
            joblib.dump(x_rain,f,compress=3)
      with open(root_path + "test/"+location+"/x_river/" + location+".jb",mode="wb") as f:
            joblib.dump(x_river,f,compress=3)
      with open(root_path + "test/"+location+"/x_geo/" + location +".jb",mode="wb") as f:
            joblib.dump(x_geo,f,compress=3)
      with open(root_path + "test/"+location+"/y_data/" + location+".jb",mode="wb") as f:
            joblib.dump(y,f,compress=3)
      geo_nc.close()
