import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d
import xarray as xr
from PIL import Image
from numba import njit

#河川固有のパラメータ, ルンゲクッタの係数
class Parameters():
    def __init__(self,dx,dy,elev,acc,dir) -> None:
        self.dt = 600
        self.dx = dx
        self.dy = dy
        self.soil_depth = np.ones_like(acc) + 0.0
        #集水面積 acc
        self.acc = acc
        #標高 elev
        self.elev = elev
        #流れの向き dir
        self.dir = np.array(dir,dtype=np.int64)
        #河道メッシュの指定
        self.riv_flag = np.where(acc>=20,1,0)
        #メッシュの面積
        self.area = dx*dy
        #メッシュの辺の長さ
        self.length = np.sqrt(self.area)
        #ルンゲクッタの係数
        self.c1 = 37/378
        self.dc1 = self.c1-2825/27648
        self.c3 = 250/621
        self.dc3 = self.c3-18575/48384
        self.c4 = 125/594
        self.dc4 = self.c4-13525/55296
        self.dc5 = -227/14336
        self.c6 = 512/1771
        self.dc6 = self.c6 - 1/4
        #段落ち式等の計算用係数
        self.mu1 = (2.0 / 3.0) ** (3.0 / 2.0)
        self.mu2 = 0.35
        self.mu3 = 0.91
        #マニングの粗度係数(ns_riv:河道部, ns_surf:斜面部)
        self.ns_riv = (np.zeros_like(acc)+0.04) #0.03
        self.ns_surf = (np.zeros_like(acc)+0.15001909) #0.4

        #土中の透水係数
        self.ka_p = (np.zeros_like(acc)+0.15)
        #green-amptの係数
        self.ksv = (np.zeros_like(acc) + 8.33/10**7)
        self.faif = (np.zeros_like(acc) + 3163 /10**5)
        self.gamma = (np.zeros_like(acc) + 0.375) #0.475
        self.infil_lim = self.gamma*self.soil_depth
        #各地点における河道幅, 深さ, メッシュあたりの河道の割合を定義
        self.set_width_depth(acc)
        #河道メッシュの最終地点(domain = 2)を定義
        self.set_domain(elev,dir)
        self.z_riv = np.where(self.riv_flag,self.elev-self.depth,self.elev)
        #土層の深さを考慮した標高
        self.z_gr = np.where(self.elev != -9999, self.elev, self.elev-self.soil_depth)

    #acc(集水範囲)から河道幅や河道深さを計算する関数
    def set_width_depth(self,acc)->None:
        width_c = 0.606#default = 5.0
        width_s = 0.80 #default = 0.35
        depth_c = 5.0 #default = 0.95
        depth_s = 0.001 #default = 0.2
        acc_temp = np.where(acc>=0,acc,0)
        width = width_c * np.array(acc_temp*dx*dy*0.000001) ** width_s
        depth = depth_c * np.array(acc_temp*dx*dy*0.000001) ** depth_s
        area_ratio = np.array(width) / self.length
        self.width = np.where(acc==-9999.0,-9999.0,width)
        self.depth = np.where(acc==-9999.0,-9999.0,depth)
        self.area_ratio = np.where(acc==-9999.0,-9999.0,area_ratio)
    
    #メッシュ毎の識別子(domain)を指定する関数
    def set_domain(self,elev,direc)->None:
        self.domain = np.zeros_like(self.riv_flag)
        for y in range(len(self.riv_flag)):
            for x in range(len(self.riv_flag[0])):
                if self.elev[y][x] != -9999.0:
                    self.domain[y][x]=1
                if self.riv_flag[y][x] != 0:
                    target = self.riv_flag[y][x]
                    if direc[y][x]==1:
                        target = self.riv_flag[y][x+1]
                    elif direc[y][x]==2:
                        target = self.riv_flag[y+1][x+1]
                    elif direc[y][x]==4:
                        target = self.riv_flag[y+1][x]
                    elif direc[y][x]==8:
                        target = self.riv_flag[y+1][x-1]
                    elif direc[y][x]==16:
                        target = self.riv_flag[y][x-1]
                    elif direc[y][x]==32:
                        target = self.riv_flag[y-1][x-1]
                    elif direc[y][x]==64:
                        target = self.riv_flag[y-1][x]
                    elif direc[y][x]==128:
                        target = self.riv_flag[y-1][x+1]
                    
                    if target != 1:
                        self.domain[y][x]=2

def read_data(project):
    elev = []
    acc =[]
    dir = []
    root = "/Users/e175764/Desktop/phy/workplace/Project/"
    path = root + project
    with open(path + "/adem.txt") as f_elev:
        elev_temp = f_elev.readlines()
    for i in range(len(elev_temp)):
        if i > 5:
            a = elev_temp[i].split()
            elev.append(list(map(float,a)))
    elev = np.array(elev)

    with open(path + "/acc.txt") as f_acc:
        acc_temp = f_acc.readlines()
    for i in range(len(acc_temp)):
        if i > 5:
            a = acc_temp[i].split()
            acc.append(list(map(float,a)))
    acc = np.array(acc)

    with open(path + "/dir.txt") as f_dir:
        dir_temp = f_dir.readlines()
    for i in range(len(dir_temp)):
        if i > 5:
            a = dir_temp[i].split()
            dir.append(list(map(float,a)))
    dir = np.array(dir)
    return elev,acc,dir

def reshape_rain(rain_list,r_shape):
    new_rains=[]
    for rain in rain_list:
        img = Image.fromarray(np.array(rain))
        img = img.resize((r_shape[0],r_shape[1]))
        new_rains.append(np.array(img))
    return new_rains

def read_rain(project,r_shape):
    root = "/Users/e175764/Desktop/phy/workplace/Project/"
    with xr.open_dataset(root+project+"/rain.nc") as rain_nc1:
        temp_rain = rain_nc1.sel(time=slice("2016-09-18T15:10:00","2016-09-21T15:00:00"))#"2012-06-18T15:00:00","2012-06-22T15:00:00"))
    r = temp_rain.rain.values
    rain_for_mesh = reshape_rain(r,r_shape)
    return rain_for_mesh
"""
表面流の計算:
    calc_surf_q_cell : 個々のメッシュ間の水の移動量,方向を計算
    calc_q_surf      : calc_surf_q_cellを用いて全体の水の移動(q)を計算
    apply_qs     : calc_q_surfで計算されたqを用いて水位を計算
2次の拡散波近似式に基づいている
"""

@njit('f8(f8[:,:],i8[:],i8[:],f8,f8,f8[:,:],f8,f8[:,:])')
def calc_surf_q_cell(h,cur,tar,dist,temp_len,elev,area,ns_surf):
    lev_cur = h[cur[1]][cur[0]] + elev[cur[1]][cur[0]]
    lev_tar = h[tar[1]][tar[0]] + elev[tar[1]][tar[0]]

    #あくまでも流出元の水位hをもとに計算
    dh = (lev_cur-lev_tar)/dist
    temp_q=0

    #going out
    if dh>=0:
        if elev[tar[1]][tar[0]] > elev[cur[1]][cur[0]]:
            hw = max(0.0,lev_cur-elev[tar[1]][tar[0]])
        hw = h[cur[1]][cur[0]]
        temp_q = np.sqrt(dh)/ns_surf[cur[1]][cur[0]] * hw ** (5.0/3.0)
    #comming in 
    else :
        if elev[tar[1]][tar[0]] < elev[cur[1]][cur[0]]:
            hw = max(0.0,lev_tar-elev[cur[1]][cur[0]])
        hw = h[tar[1]][tar[0]]
        dh = np.abs(dh)
        temp_q = -np.sqrt(dh)/ns_surf[tar[1]][tar[0]] * hw ** (5.0/3.0)
    return temp_q*temp_len/area

#domain,elev,ns_surf,dx,dy
@njit('f8[:,:,:](f8[:,:],i8[:,:],f8[:,:],f8[:,:],f8,f8)')
def calc_q_surf(h,domain,elev,ns_surf,dx,dy): 
    #temp_q = [[0.0,0.0,0.0,0.0] for i in range(len(h[0]))] #4方向なので0,0,0,0 
    #q = [temp_q for i in range(len(h))]
    q = np.zeros((len(h),len(h[0]),4),dtype=np.float64)
    #h = h.tolist()
    area = dx*dy
    for y in range(len(h)):
        for x in range(len(h[0])):
            if domain[y][x] == 0: continue
            #in:- out:+
            dist = np.sqrt(dx**2 +dy**2)
            cur = np.array([x,y],dtype=np.int64)
            tar = np.array([x,y],dtype=np.int64)

            #右
            dist = dx
            temp_len = dy/2
            tar = np.array([x+1,y],dtype=np.int64)
            if domain[tar[1]][tar[0]] !=0 :
                q[y][x][0]=calc_surf_q_cell(h,cur,tar,dist,temp_len,elev,area,ns_surf)

            #右下
            dist = np.sqrt(dx**2 + dy**2)
            tar = np.array([x+1,y+1],dtype=np.int64)
            temp_len = dist/4
            if domain[tar[1]][tar[0]] != 0:
                q[y][x][1]=calc_surf_q_cell(h,cur,tar,dist,temp_len,elev,area,ns_surf)
            
            #下
            dist = dy
            tar = np.array([x,y+1],dtype=np.int64)
            temp_len = dx/2
            if domain[tar[1]][tar[0]] != 0:
                q[y][x][2]=calc_surf_q_cell(h,cur,tar,dist,temp_len,elev,area,ns_surf)

            #左下
            if x-1 >=0:
                dist = np.sqrt(dx**2 + dy**2)
                tar = np.array([x-1,y+1],dtype=np.int64)
                temp_len=dist/4
                if domain[tar[1]][tar[0]] != 0:
                    q[y][x][3]=calc_surf_q_cell(h,cur,tar,dist,temp_len,elev,area,ns_surf)
    return q
@njit('f8[:,:](f8[:,:],f8[:,:,:],i8[:,:])')
def apply_qs(r,q,domain):
    #apply_q
    fs = np.zeros_like(r,dtype=np.float64)
    fs = np.where(domain!=0,r-np.sum(q,axis=2),0.0)
    for y in range(len(r)):
        for x in range(len(r[0])):
            #各方面への流出を計算(各方面からの流入でもある)
            if domain[y][x] == 0: continue
            #fs[y][x] = fs[y][x] + r - np.sum(q[y][x]) 
            #右(符号注意)
            fs[y][x+1]+=q[y][x][0]
            #右下(符号注意)
            fs[y+1][x+1]+=q[y][x][1]
            #下(符号注意)
            fs[y+1][x]+=q[y][x][2]
            #左下
            fs[y+1][x-1]+=q[y][x][3]
    return fs


"""
河道流の計算:
    calc_riv_q_cell : 個々のメッシュ間の水の移動量,方向を計算
    calc_q_riv      : calc_riv_q_cellを用いて全体の水の移動(q)を計算
    apply_qr     : calc_q_rivで計算されたqを用いて水位を計算
表面水位とは関係なく,河道水位の変動のみを追従
"""
#z_riv,ns_riv,width
@njit('f8(f8[:,:],f8[:,:],f8[:,:],f8[:,:],i8[:,:],i8[:],i8[:],f8)')
def calc_riv_q_cell(h,z_riv,ns_riv,width,domain,cur,tar,dist):
    lev_cur = h[cur[1]][cur[0]] + z_riv[cur[1]][cur[0]]
    lev_tar = h[tar[1]][tar[0]] + z_riv[tar[1]][tar[0]]
    if domain[tar[1]][tar[0]] == 2: #if distination cell is outlet(not river)
        lev_tar = z_riv[tar[1]][tar[0]]
    
    dh = (lev_cur-lev_tar)/dist
    temp_q=0
    #going out

    if dh>=0:
        hw = h[cur[1]][cur[0]]
        if z_riv[tar[1]][tar[0]] > z_riv[cur[1]][cur[0]]:
            hw = max(0.0,lev_cur-z_riv[tar[1]][tar[0]])
        temp_q = np.sqrt(dh)/ns_riv[cur[1]][cur[0]] * hw ** (5.0/3.0) * width[cur[1]][cur[0]]
    #comming in 
    else :
        hw = h[tar[1]][tar[0]]
        if z_riv[tar[1]][tar[0]] < z_riv[cur[1]][cur[0]]:
            hw = max(0.0,lev_tar-z_riv[cur[1]][cur[0]])
        dh = np.abs(dh)
        temp_q = -np.sqrt(dh)/ns_riv[tar[1]][tar[0]] * hw ** (5.0/3.0) * width[cur[1]][cur[0]]
    return temp_q

#vr,dx,dy,area_ratio,riv_frag,dir,z_riv,ns_riv,width
@njit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],i8[:,:],i8[:,:],i8[:,:],f8,f8)')
def calc_q_riv(vr,area_ratio,z_riv,ns_riv,width,domain,riv_flag,direc,dx,dy):
    area = dx*dy
    #q = [[0 for i in range(len(vr[0]))] for j in range(len(vr))]#np.zeros_like(vr)+0.0
    q = np.zeros_like(vr,dtype=np.float64)
    #hr = np.where(np.array(param.riv_flag),np.array(vr),0) / (param.area * np.array(param.area_ratio))
    hr = np.zeros_like(vr,dtype=np.float64)
    for y in range(len(hr)):
        for x in range(len(hr[0])):
            if riv_flag[y][x] != 1: continue
            hr[y][x] = vr[y][x]/(area*area_ratio[y][x])
            cur = [x,y]
            tar = [x,y]
            dist = np.sqrt(dx**2 + dy**2)
            temp_len = area/np.sqrt(dx**2 + dy**2)
            if direc[y][x]==1:#右
                tar = [x+1,y]
                dist = dx
                temp_len = dy
            elif direc[y][x]==2:#右下
                tar = [x+1,y+1]
            elif direc[y][x]==4:#下
                tar = [x,y+1]
                dist = dy
                temp_len = dx
            elif direc[y][x]==8:#左下
                tar = [x-1,y+1]
            elif direc[y][x]==16:#左
                tar = [x-1,y]
                dist = dx
                temp_len = dy
            elif direc[y][x]==32:#左上
                tar = [x-1,y-1]
            elif direc[y][x]==64:#上
                tar = [x,y-1]
                dist = dy
                temp_len = dx
            elif direc[y][x]==128:#右上
                tar = [x+1,y-1]
            q[y][x] = calc_riv_q_cell(hr,z_riv,ns_riv,width,domain,np.array(cur,dtype=np.int64),np.array(tar,dtype=np.int64),dist)
    return q

@njit('f8[:,:](f8[:,:],i8[:,:],i8[:,:])')
def apply_qr(q,direc,riv_flag):
    fr = np.zeros_like(q,dtype=np.float64)
    #np.where(param.riv_flag,-np.array(q),0.0).tolist()
    for y in range(len(direc)):
        for x in range(len(direc[0])):
            #河道部のみを計算
            if riv_flag[y][x] == 0 : continue
            #fr[y][x] -= q[y][x]
            fr[y][x] -= q[y][x]
            #各方面への流出を計算(各方面からの流入でもある)
            if direc[y][x]==1:#右
                fr[y][x+1]+=q[y][x]
            elif direc[y][x]==2:#右下
                fr[y+1][x+1]+=q[y][x]
            elif direc[y][x]==4:#下
                fr[y+1][x]+=q[y][x]
            elif direc[y][x]==8:#左下
                fr[y+1][x-1]+=q[y][x]
            elif direc[y][x]==16:#左
                fr[y][x-1]+=q[y][x]
            elif direc[y][x]==32:#左上
                fr[y-1][x-1]+=q[y][x]
            elif direc[y][x]==64:#上
                fr[y-1][x]+=q[y][x]
            elif direc[y][x]==128:#右上
                fr[y-1][x+1]+=q[y][x]
    return fr

"""
不飽和側方流の計算
    calc_gr_q_cell : 個々のメッシュ間の水の移動量,方向を計算
    calc_q_gr      : calc_riv_q_cellを用いて全体の水の移動(q)を計算
    apply_qg    : calc_q_rivで計算されたqを用いて水位を計算
"""
@njit('f8(f8[:,:],f8[:,:],f8[:,:],i8[:],i8[:],f8,f8,f8)')
def calc_gr_q_cell(h,z_gr,ka_p,cur,tar,dist,temp_len,area):
    lev_cur = h[cur[1]][cur[0]] + z_gr[cur[1]][cur[0]]
    lev_tar = h[tar[1]][tar[0]] + z_gr[tar[1]][tar[0]]    
    
    dh = (lev_cur-lev_tar)/dist
    temp_q=0

    #going out
    if dh>=0:
        hw = h[cur[1]][cur[0]]
        if z_gr[tar[1]][tar[0]] > z_gr[cur[1]][cur[0]]:
            hw = max(0.0,lev_cur-z_gr[tar[1]][tar[0]])
        temp_q = ka_p[cur[1]][cur[0]] * dh * hw
    #comming in 
    else :
        hw = h[tar[1]][tar[0]]
        if z_gr[tar[1]][tar[0]] < z_gr[cur[1]][cur[0]]:
            hw = max(0.0,lev_tar-z_gr[cur[1]][cur[0]])
        dh = np.abs(dh)
        temp_q = -ka_p[cur[1]][cur[0]] * dh * hw
    return temp_q*temp_len/area

@njit('f8[:,:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:],i8[:,:],f8,f8)')
def calc_q_gr(h,z_gr,ka_p,elev,domain,dx,dy):
    #temp_q = [[0.0,0.0,0.0,0.0] for i in range(len(h[0]))] #4方向なので0,0,0,0 
    #q = [temp_q for i in range(len(h))]
    area = dx*dy
    q = np.zeros((len(h),len(h[0]),4),dtype=np.float64)
    for y in range(len(h)):
        for x in range(len(h[0])):
            if domain[y][x] == 0: continue
            #in:- out:+
            dist = np.sqrt(dx**2 +dy**2)
            cur = [x,y]
            tar = [x,y]
            temp_h = h[y][x]

            #右
            dist = dx
            temp_len = dy/2
            tar = [x+1,y]
            if domain[tar[1]][tar[0]] !=0 :
                q[y][x][0]=calc_gr_q_cell(h,z_gr,ka_p,np.array(cur,dtype=np.int64),np.array(tar,dtype=np.int64),dist,temp_len,area)

            #右下
            dist = np.sqrt(dx**2 + dy**2)
            tar = [x+1,y+1]
            temp_len = dist/4
            if domain[tar[1]][tar[0]] != 0:
                q[y][x][1]=calc_gr_q_cell(h,z_gr,ka_p,np.array(cur,dtype=np.int64),np.array(tar,dtype=np.int64),dist,temp_len,area)
            
            #下
            dist = dy
            tar = [x,y+1]
            temp_len = dx/2
            if domain[tar[1]][tar[0]] != 0:
                q[y][x][2]=calc_gr_q_cell(h,z_gr,ka_p,np.array(cur,dtype=np.int64),np.array(tar,dtype=np.int64),dist,temp_len,area)

            #左下
            if x-1 >=0:
                dist = np.sqrt(dx**2 + dy**2)
                tar = [x-1,y+1]
                temp_len=dist/4
                if domain[tar[1]][tar[0]] != 0:
                    q[y][x][3]=calc_gr_q_cell(h,z_gr,ka_p,np.array(cur,dtype=np.int64),np.array(tar,dtype=np.int64),dist,temp_len,area)
    return q

@njit('f8[:,:](f8[:,:,:],f8[:,:],i8[:,:],i8[:,:])')
def apply_qg(q,gamma,riv_flag,domain):
    #fs = np.zeros_like(param.riv_flag) + 0.0
    #apply_q
    #流出した分を足す
    fg = np.where(domain!=0,-np.sum(q,axis=2)/gamma,0.0)
    for y in range(len(riv_flag)):
        for x in range(len(riv_flag[0])):
            #各方面への流出を計算(各方面からの流入でもある)
            if domain[y][x] == 0: continue
            #右(符号注意)
            fg[y][x+1]+=q[y][x][0]
            #右下(符号注意)
            fg[y+1][x+1]+=q[y][x][1]
            #下(符号注意)
            fg[y+1][x]+=q[y][x][2]
            #左下
            if x-1 >=0:
                fg[y+1][x-1]+=q[y][x][3]
    return fg


"""
斜面-河道の計算
斜面部水位(hs)と河道水位(hr)の水のやり取り
表面流出と復帰流出の計算部を分けたい
"""
def surf_riv_ex(hs,hr,param,txt_path,t):
    hs_new = np.zeros_like(hs) + 0.0
    hr_new = np.zeros_like(hr) + 0.0
    all_hrs = []
    for y in range(len(hs)):
        for x in range(len(hs[0])):
            hs_new[y][x] = hs[y][x]
            hr_new[y][x] = hr[y][x]
            if param.domain[y][x] == 0: continue
            if param.riv_flag[y][x] == 0: continue
            hs_top = hs[y][x]
            hr_top = hr[y][x] - param.depth[y][x]
            
            if hr_top<=0:
                #面積あたりの流出高
                hrs = param.mu1 * hs_top * np.sqrt(9.81*hs_top) * param.dt * param.length / param.area
                #流出量 > 表面水位 : 全部川へ
                if hrs > hs_top: hrs = hs_top 
                #流出後の表面及び地中水位
                hs_new[y][x] -= hrs
                #流出高 -> 流出量 and 水位 -> 流量
                temp_vrs = hrs * param.area
                temp_v = hr[y][x] * param.area * param.area_ratio[y][x]
                #流量+=流出量 and 流量->水位
                temp_v += temp_vrs
                hr_new[y][x] = temp_v / (param.area * param.area_ratio[y][x])
                all_hrs.append(hrs)
    np.savetxt(txt_path+"hrs_"+str(t)+".csv",all_hrs,delimiter=',')
    return hs_new,hr_new

"""
鉛直浸透の計算
    infiltration: 表面水深->地中水深
    exfiltration: 地中水深の溢れた分 -> 表面水深
"""
def infiltration(hs,hg,total_infil,param):
    total_infil=np.where(total_infil <= 0.01,0.01,total_infil)
    temp_infil = np.zeros_like(hs) + 0.0
    temp_infil = param.ksv * (1.0 + param.faif * param.gamma / total_infil)
    temp_infil = np.where(temp_infil >= hs/param.dt, hs/param.dt, temp_infil)
    temp_infil = np.where(total_infil >= param.infil_lim ,0 ,temp_infil)
    temp_infil = np.where(hg >= param.soil_depth,0,temp_infil)
    total_infil += temp_infil * param.dt
    #update hs and hg using infiltartion rate * dt
    #temp
    hs_new = hs - temp_infil*dt
    hg_new = hg + (temp_infil*dt) / param.gamma
    hs_new = np.where(hs_new<0,0,hs_new)
    return hs_new,hg_new,total_infil

def exfiltration(hs,hg,param):
    exfil = np.where(hg >= param.soil_depth,hg-param.soil_depth,0)
    #exfil -> dhg
    #dhg = dhs/gamma
    hs_new = hs + (exfil * param.gamma)
    hg_new = hg - exfil
    return hs_new,hg_new

def view_h(t,h,param,project,folder,max_val):
    ph = np.where(param.elev==-9999.0,-9999.0, h)
    root = "/Users/e175764/Desktop/phy/workplace/Project/"
    path = root + project + "/result/" + folder + "/"
    plt.figure()
    plt.imshow(ph,vmin=0,vmax=max_val,cmap='jet')
    plt.colorbar()
    plt.savefig(path + str(t) +".png")
    plt.close()

def create_init_depth(w_lev,w_loc,param):
    under_lim_y = 7
    over_lim_y = 43
    init_hr = np.zeros_like(param.riv_flag) + 0.0
    pp = np.where(acc>200)
    main = [[pp[0][i],pp[1][i]] for i in range(len(pp[0]))]
    linespace_x = [i for i in range(len(main))]
    fx=[]
    fz=[]
    for i in range(len(main)):
        for j in range(len(w_loc)):
            if main[i] == w_loc[j]:
                fx.append(i)
                fz.append(w_lev[j])
    f = interp1d(fx,fz,kind="linear")
    for y in range(len(param.riv_flag)):
        for x in range(len(param.riv_flag[0])):
            if param.riv_flag[y][x] == 0: continue
            if [y,x] in main:
                for idx in range(len(main)):
                    if [y,x]==main[idx]:break
                try:
                    init_hr[y][x] = f(idx)
                except:
                    pass
    #ここまで(集水面積の大きいところの補間)を基にyの値を作成？
    center_y = sum(fx)/len(fx)
    fxx = list(fx)
    fzz = list(fz)
    fxx.insert(0,under_lim_y)
    fxx.append(over_lim_y)
    fzz.insert(0,0.05)
    fzz.append(0.05)
    fy = interp1d(fxx,fzz,kind="linear")
    for y in range(len(param.riv_flag)):
        if over_lim_y > y > under_lim_y:
            max_val = max(init_hr[y])
            max_idx = list(init_hr[y]).index(max_val)
            if max_val == 0:
                max_val = fy(y)
                max_idx = center_y
            f = interp1d([8,max_idx,48],[0.05,max_val,0.05],kind="linear")
            for x in range(len(param.riv_flag[0])):
                if param.riv_flag[y][x] == 0: continue
                init_hr[y][x] = f(x)
    return init_hr

def calc_total_water(hs,hr,hg,param):
    total_water = 0
    total_water += (np.where(param.domain!=0,hs,0) * param.area).sum()
    total_water += (np.where(param.riv_flag,hr,0)*param.area_ratio * param.area).sum()
    total_water += (np.where(param.domain!=0,hg,0)*param.gamma * param.area).sum()
    return total_water

def check_water_balance(total_water,r,hs,hr,hg,out,param):
    aft_total=0
    rainfall = (np.where(param.domain!=0,r,0) * param.area).sum()
    aft_total += (np.where(param.domain!=0,hs,0) * param.area).sum()
    aft_total += (np.where(param.riv_flag,hr,0)*param.area_ratio * param.area).sum()
    aft_total += (np.where(param.domain!=0,hg,0)*param.gamma * param.area).sum()
    balance = (total_water + rainfall) - (aft_total + out)
    if balance < 10**(-8):
        print("water_balance ok")
    return 0

def get_init_h(proj,txt,time):
    txt_hs = txt + "hs_" + str(time) + ".csv"
    txt_hr = txt + "hr_" + str(time) + ".csv"
    txt_hg = txt + "hg_" + str(time) + ".csv"
    hs = np.loadtxt(txt_hs,delimiter=",")
    hr = np.loadtxt(txt_hs,delimiter=",")
    hg = np.loadtxt(txt_hs,delimiter=",")
    return hs,hr,hg

if __name__=="__main__":
    proj = "test/20160919"
    root = "/Users/e175764/Desktop/phy/workplace/Project/"
    txt_path = root + proj + "/result/result_csv/"
    elev,acc,dir = read_data("test")
    w_loc = [[15,34],[24,29],[26,28],[31,27]]
    w_lev = [0.61,0.10,0.99,0.1]#[1.63,0.75,1.56,0.1] [1.61,0.66,1.32,0.1] 2011:[1.45,0.58,1.42,0.1] 2016/06[0.57,0.10,0.87,0.1]
    #elev = np.array([[float(i)*100,float(i+1)*100,float(i+2)*100] for i in range(3)]) #標高
    hs = np.where(elev!=-9999.0,0.0,0.0)/1000 #表面水位
    hg = np.zeros_like(elev) + 0.0
    total_infil = np.zeros_like(elev) + 0.0
    r = read_rain(proj,[57,51])
    #r += 36
    dx = 919.782631617873
    dy = 921.607878503749
    params = Parameters(dx,dy,elev,acc,dir)
    #hr = np.where(params.riv_flag,0.1,0) + 0.0 #河川水位
    #12-35あたりを見れば良い？
    dt = 600
    max_t = 3*3600/dt
    ddt_r_all = np.array(r)/(3600*1000) #1秒あたりの降雨(m/s)
    #print(ddt_r)
    #view_h(0,params.domain,elev,proj,2)
    count = 0
    out = 0
    #hs = np.where(np.array(params.domain)!=0,1.0,0.0)
    hr = create_init_depth(w_lev,w_loc,params)
    #view_h(0,hr,elev,proj,2)
    """
    for y in range(len(params.riv_flag)):
        for x in range(len(params.riv_flag[0])):
            if params.riv_flag[y][x]:
                print([y,x])
    """
    #216001
    #hs * gamma = hg 
    for t in range(count*dt+1,count*dt+216001,dt):
        ddt_r = ddt_r_all[count]
        ddt_r = np.where(ddt_r>=0,ddt_r,0)
        #ddt_r = np.zeros_like(ddt_r)
        #view_h(t,ddt_r,params,proj,"rain",0.00006)
        #view_h(t,hs,params,proj,"surf",2.0)
        #view_h(t,hr,params,proj,"river",10.0)
        #view_h(t,hg,params,proj,"ground",1.0)
        total_water = calc_total_water(hs,hr,hg,params)
        #view_h(t,hg,params,proj,"temp",0.3)
        #1.総降雨量 r*dt*area * num(domain!=0) [m^3]
        #2.総表面水量 hs * area * num(domain != 0) [m^3]
        #3.総河道水量 hr * area * area_ratio [m^3]
        #4.総地中水量 hg * area * gamma (domain != 0) [m^3]
        #5.境界水量 out [m^3]

        #river flow calculation
        time = (t-1)*dt
        ddt = dt/10
        vr = hr * params.area * np.array(params.area_ratio)
        pre_vr = vr.copy()
        while(1):
            #vr,area_ratio,z_riv,ns_riv,width,domain,riv_flag,dir,dx,dy
            if(time+ddt > t*dt): ddt=t*dt-time
            q = calc_q_riv(vr,params.area_ratio,params.z_riv,params.ns_riv,params.width,params.domain,params.riv_flag,params.dir,dx,dy)
            fr = apply_qr(q,params.dir,params.riv_flag)
            vr_temp = vr + (1/5)*ddt*fr
            vr_temp = np.where(np.array(vr_temp) >= 0,np.array(vr_temp),0)

            #(2)
            q = calc_q_riv(vr_temp,params.area_ratio,params.z_riv,params.ns_riv,params.width,params.domain,params.riv_flag,params.dir,dx,dy)
            fr_2 = apply_qr(q,params.dir,params.riv_flag)
            vr_temp = vr + (1/40)*ddt*(3*fr+9*fr_2)
            vr_temp = np.where(np.array(vr_temp) >= 0,np.array(vr_temp),0)
            
            #(3)
            q = calc_q_riv(vr_temp,params.area_ratio,params.z_riv,params.ns_riv,params.width,params.domain,params.riv_flag,params.dir,dx,dy)
            fr_3 = apply_qr(q,params.dir,params.riv_flag)
            vr_temp = vr + (1/10)*ddt*(3*fr-9*fr_2+12*fr_3)
            vr_temp = np.where(np.array(vr_temp) >= 0,np.array(vr_temp),0)

            #(4)
            q = calc_q_riv(vr_temp,params.area_ratio,params.z_riv,params.ns_riv,params.width,params.domain,params.riv_flag,params.dir,dx,dy)
            fr_4 = apply_qr(q,params.dir,params.riv_flag)
            vr_temp = vr + (1/54)*ddt*(-11*fr+135*fr_2-140*fr_3+70*fr_4)
            vr_temp = np.where(np.array(vr_temp) >= 0,np.array(vr_temp),0)

            #(5)
            q = calc_q_riv(vr_temp,params.area_ratio,params.z_riv,params.ns_riv,params.width,params.domain,params.riv_flag,params.dir,dx,dy)
            fr_5 = apply_qr(q,params.dir,params.riv_flag)
            vr_temp = vr + (1/110592)*ddt*(3262*fr+37800*fr_2+4600*fr_3+44275*fr_4+6831*fr_5)
            vr_temp = np.where(np.array(vr_temp) >= 0,np.array(vr_temp),0)

            #(6)
            q = calc_q_riv(vr_temp,params.area_ratio,params.z_riv,params.ns_riv,params.width,params.domain,params.riv_flag,params.dir,dx,dy)
            fr_6 = apply_qr(q,params.dir,params.riv_flag)
            vr_temp = vr + ddt*(params.c1*fr + params.c3*fr_3 +params.c4*fr_4 +params.c6*fr_6)
            vr_temp = np.where(np.array(vr_temp) >= 0,np.array(vr_temp),0)
            #(e)
            vr_error = ddt*(params.dc1*fr + params.dc3*fr_3 + params.dc4*fr_4 + params.dc5*fr_5 + params.dc6*fr_6)
            hr_error = np.where(params.riv_flag,vr_error/(params.area*np.array(params.area_ratio)),0)

            error_max = np.max(np.array(hr_error))/0.01
            #実際にはこれでいいのでは？刻み幅とtimeの挙動が気になる
            
            #if error_max <=1 :
            #    h = hs_temp
            #    break

            #以下の条件文の精査
            #比較的単純な例を用いた実装の検証
            if error_max>1 and ddt > 1:
                prev = ddt
                ddt=max(0.9*ddt*(error_max**(-0.25)),0.5*ddt)
                if np.isnan(error_max):
                    ddt = 0.5*prev
            else:
                if time+ddt > t*dt:
                    ddt=t*dt-time
                time = time+ddt
                vr=vr_temp
            if time>=t*dt : break
        hr = vr / (params.area*np.array(params.area_ratio))
        #if abs(pre_vr.sum() - vr.sum()) <0.00001 :
        #    print("river ok")

        #surface flow calculation
        #domain,elev,ns_surf,dx,dy
        pre_hs = hs.copy()
        time = (t-1)*dt
        ddt = dt
        while(1):
            if(time+ddt > t*dt): ddt=t*dt-time
            #(1)
            q = calc_q_surf(hs,params.domain,elev,params.ns_surf,dx,dy)
            fs = apply_qs(ddt_r,q,params.domain)
            hs_temp = hs + (1/5)*ddt*fs
            hs_temp = np.where(np.array(hs_temp) >= 0,np.array(hs_temp),0)

            #(2)
            q = calc_q_surf(hs_temp,params.domain,elev,params.ns_surf,dx,dy)
            fs_2 = apply_qs(ddt_r,q,params.domain)
            hs_temp = hs + (1/40)*ddt*(3*fs+9*fs_2)
            hs_temp = np.where(np.array(hs_temp) >= 0,np.array(hs_temp),0)
            
            #(3)            
            q = calc_q_surf(hs_temp,params.domain,elev,params.ns_surf,dx,dy)
            fs_3 = apply_qs(ddt_r,q,params.domain)
            hs_temp = hs + (1/10)*ddt*(3*fs-9*fs_2+12*fs_3)
            hs_temp = np.where(np.array(hs_temp) >= 0,np.array(hs_temp),0)

            #(4)
            q = calc_q_surf(hs_temp,params.domain,elev,params.ns_surf,dx,dy)
            fs_4 = apply_qs(ddt_r,q,params.domain)
            hs_temp = hs + (1/54)*ddt*(-11*fs+135*fs_2-140*fs_3+70*fs_4)
            hs_temp = np.where(np.array(hs_temp) >= 0,np.array(hs_temp),0)

            #(5)
            q = calc_q_surf(hs_temp,params.domain,elev,params.ns_surf,dx,dy)
            fs_5 = apply_qs(ddt_r,q,params.domain)
            hs_temp = hs + (1/110592)*ddt*(3262*fs+37800*fs_2+4600*fs_3+44275*fs_4+6831*fs_5)
            hs_temp = np.where(np.array(hs_temp) >= 0,np.array(hs_temp),0)

            #(6)
            q = calc_q_surf(hs_temp,params.domain,elev,params.ns_surf,dx,dy)
            fs_6 = apply_qs(ddt_r,q,params.domain)
            hs_temp = hs + ddt*(params.c1*fs + params.c3*fs_3 +params.c4*fs_4 +params.c6*fs_6)
            hs_temp = np.where(np.array(hs_temp) >= 0,np.array(hs_temp),0)

            #(e)
            hs_error = ddt*(params.dc1*fs + params.dc3*fs_3 + params.dc4*fs_4 + params.dc5*fs_5 + params.dc6*fs_6)
            hs_error = np.where(params.domain!=0,hs_error,0)

            error_max = np.max(np.array(hs_error))/0.01

            if error_max>1 and ddt > 1:
                prev = ddt
                ddt=max(0.9*ddt*(error_max**(-0.25)),0.5*ddt)
                if np.isnan(error_max):
                    ddt = 0.5*prev
            else:
                if time+ddt > t*dt:
                    ddt=t*dt-time
                time = time+ddt
                hs=hs_temp
            if time>=t*dt : break
        #if abs((pre_hs.sum()+np.where(params.domain!=0,ddt_r,0).sum()*dt) - hs.sum()) < 0.0001:
        #    print("surf ok")



        #Grand water calculation
        #h,z_gr,ka_p,elev,domain,dx,dy
        pre_hg = hg.copy()
        time = (t-1)*dt
        ddt = dt
        while(1):
            if(time+ddt > t*dt): ddt=t*dt-time
            #(1)
            qg = calc_q_gr(hg,params.z_gr,params.ka_p,params.elev,params.domain,dx,dy)
            fg = apply_qg(qg,params.gamma,params.riv_flag,params.domain)
            hg_temp = hg + (1/5)*ddt*fg
            hg_temp = np.where(np.array(hg_temp) >= 0,np.array(hg_temp),0)

            #(2)
            qg = calc_q_gr(hg_temp,params.z_gr,params.ka_p,params.elev,params.domain,dx,dy)
            fg_2 = apply_qg(qg,params.gamma,params.riv_flag,params.domain)
            hg_temp = hg + (1/40)*ddt*(3*fg+9*fg_2)
            hg_temp = np.where(np.array(hg_temp) >= 0,np.array(hg_temp),0)
            
            #(3)
            qg = calc_q_gr(hg_temp,params.z_gr,params.ka_p,params.elev,params.domain,dx,dy)
            fg_3 = apply_qg(qg,params.gamma,params.riv_flag,params.domain)
            hg_temp = hg + (1/10)*ddt*(3*fg-9*fg_2+12*fg_3)
            hg_temp = np.where(np.array(hg_temp) >= 0,np.array(hg_temp),0)

            #(4)
            qg = calc_q_gr(hg_temp,params.z_gr,params.ka_p,params.elev,params.domain,dx,dy)
            fg_4 = apply_qg(qg,params.gamma,params.riv_flag,params.domain)
            hg_temp = hg + (1/54)*ddt*(-11*fg+135*fg_2-140*fg_3+70*fg_4)
            hg_temp = np.where(np.array(hg_temp) >= 0,np.array(hg_temp),0)

            #(5)
            qg = calc_q_gr(hg_temp,params.z_gr,params.ka_p,params.elev,params.domain,dx,dy)
            fg_5 = apply_qg(qg,params.gamma,params.riv_flag,params.domain)
            hg_temp = hg + (1/110592)*ddt*(3262*fg+37800*fg_2+4600*fg_3+44275*fg_4+6831*fg_5)
            hg_temp = np.where(np.array(hg_temp) >= 0,np.array(hg_temp),0)

            #(6)
            qg = calc_q_gr(hg_temp,params.z_gr,params.ka_p,params.elev,params.domain,dx,dy)
            fg_6 = apply_qg(qg,params.gamma,params.riv_flag,params.domain)
            hg_temp = hg + ddt*(params.c1*fg + params.c3*fg_3 +params.c4*fg_4 +params.c6*fg_6)
            hg_temp = np.where(np.array(hg_temp) >= 0,np.array(hg_temp),0)

            #(e)
            hg_error = ddt*(params.dc1*fg + params.dc3*fg_3 + params.dc4*fg_4 + params.dc5*fg_5 + params.dc6*fg_6)
            hg_error = np.where(params.domain,hg_error,0)

            error_max = np.max(np.array(hg_error))/0.01
            if error_max>1 and ddt > 1:
                prev = ddt
                ddt=max(0.9*ddt*(error_max**(-0.25)),0.5*ddt)
                if np.isnan(error_max):
                    ddt = 0.5*prev
            else:
                if time+ddt > t*dt:
                    ddt=t*dt-time
                time = time+ddt
                hg=hg_temp
            if time>=t*dt : break
        #if abs(pre_hg.sum() - hg.sum()) < 0.00001:
        #    print("ground ok")
             
        #exfiltration
        hg = np.where(params.domain!=0,hg,0.0)
        pre_hg = hg.copy() #hs基準の水量(m)?
        pre_hs = hs.copy()
        hs,hg = exfiltration(hs,hg,params)
        #hg - > hr : dhg/porosity = dhs
        #if np.allclose((hs-pre_hs)*params.gamma,pre_hg-hg):
        #    print("exfil ok")


        #surface to river (hs -> hr)
        pre_hs = hs.copy()
        pre_vr = hr * params.area * params.area_ratio #[m^3]
        hs,hr = surf_riv_ex(hs,hr,params,txt_path,t)
        aft_vr = hr * params.area * np.array(params.area_ratio)
        #if np.allclose((pre_hs - hs)*params.area,aft_vr-pre_vr):
        #    print("exchange ok")

        #infiltration
        pre_hs = hs.copy()
        pre_hg = hg.copy()
        hs,hg,total_infil = infiltration(hs,hg,total_infil,params)
        #hs -> hg : dhg = dhs / porosity 
        #if np.allclose((pre_hs-hs)/params.gamma,(hg-pre_hg)):
        #    print("infil ok")

        temp_total = calc_total_water(hs,hr,hg,params)
        temp_balance = temp_total - (total_water+(np.where(params.domain!=0,ddt_r,0)*dt*params.area).sum())
        #if temp_balance == 0:
        #    print("temp total ok")
        
        #outlet(boundary) calculation
        q_out = np.where(params.domain==2,hr,0) * params.area * np.array(params.area_ratio)
        hr = np.where(params.domain==2,0,hr)
        hs_out = np.where(params.domain==2,hs,0) * params.area
        hs = np.where(params.domain==2,0,hs)
        hg_out = np.where(params.domain==2,hg,0) * params.area * params.gamma
        hg = np.where(params.domain==2,0,hg)
        out = hs_out.sum() + hg_out.sum() + q_out.sum()
        check_water_balance(total_water,ddt_r*dt,hs,hr,hg,out,params)
        
        #np.savetxt(txt_path+"ti_"+str(t)+".csv",total_infil,delimiter=',')
        #np.savetxt(txt_path+"hs_"+str(t)+".csv",hs,delimiter=',')
        np.savetxt(txt_path+"hr_"+str(t)+".csv",hr,delimiter=',')
        #np.savetxt(txt_path+"hg_"+str(t)+".csv",hg,delimiter=',')
        count += 1
        