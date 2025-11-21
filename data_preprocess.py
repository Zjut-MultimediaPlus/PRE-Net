import logging
import warnings
warnings.filterwarnings('ignore')
import os
import re
import netCDF4 as nc
import numpy as np
import copy
from osgeo import gdal,gdalconst,osr
import matplotlib.pyplot as plt

import xarray as xr


class Preprocess(object):

    def __init__(self,primaryfile = None, outpath = None ):
        '''
        :param primaryfile:  带原始文件
        :param outpath: 输出路径
        '''
        self.primaryfile = primaryfile
        self.outpath = outpath

        self.disk_line_number = 2748
        self.disk_pixel_number = 2748
        self.L1Channel= {"01": "VIS047",
                         "02":"VIS065",
                         "03": "NIR083",
                         "04":"SIR137",
                         "05": "SIR161",
                         "06":"SIR222",
                         "07": "MIR372H",
                         "08":"MIR372L",
                         "09": "VAP625",
                         "10":"VAP710",
                         "11": "IR085",
                         "12":"IR108",
                         "13": "IR120",
                         "14":"IR135"}

        if "2000M" in  self.primaryfile and "GHI-" in self.primaryfile:

            self.disk_line_number,self.disk_pixel_number=5496,5496
        elif "500M" in self.primaryfile and "GHI-" in self.primaryfile:
            self.disk_line_number, self.disk_pixel_number = 21984, 21984



    def Read_FY4_Channel(self,filepath,Channel='01'):
        f = nc.Dataset(filepath,'r')

        if Channel =="07" and "FY4A" in filepath:
            sh = 65534
        else:
            sh = 4096
        try:
            NOMChannel = f['NOMChannel%s'%(Channel)][:]
            CALChannel = f['CALChannel%s'%(Channel)][:]
        except:
            NOMChannel = f["Data"]['NOMChannel%s'%(Channel)][:]
            CALChannel = f["Calibration"]['CALChannel%s'%(Channel)][:]
        Channel_data = self.Data_Cal(NOMChannel, CALChannel, sh)

        begin_line = f.getncattr("Begin Line Number")
        end_line = f.getncattr("End Line Number")

        f.close()
        if 'DISK' in filepath:
            Channel_data = Channel_data
        elif 'REGC' in filepath:
            temparray = np.full((self.disk_line_number, self.disk_pixel_number),0.0)
            temparray[begin_line:end_line+1, :] = Channel_data
            Channel_data = temparray

        return Channel_data.astype(np.float64)

    def FY4_reproject(self,data_path,datavar, llat=-5, ulat=50, llon=65, ulon=140,res=0.04):
        """
        几何校正
        :param data_path:
        :return:0,正确
                1,有问题
        """

        if os.path.isfile(data_path):
            (filepath, fy4_filename) = os.path.split(data_path)
            fy4_filename = str.strip(fy4_filename)
        else:
            return 1, "file_path is not a file!!!"

        temp = 'temp'
        if os.path.isdir(temp) == False:
            os.mkdir(temp)
        # 分辨率为4km
        res0 = int(data_path.split('_')[-2][:-1])

        datatype = gdal.GDT_Float32  # 16位
        dstFilePath = os.path.join(temp, fy4_filename[:-3]) + 'tif'

        Driver = gdal.GetDriverByName('MEM')
        memDs = Driver.Create('', self.disk_line_number, self.disk_pixel_number, 1, datatype)

        srs = osr.SpatialReference()
        if "FY4A" in data_path:
            srs.ImportFromProj4('+proj=geos +h=35785863 +a=6378137.0 +b=6356752.3 +lon_0=104.7 +no_defs')
        elif "FY4B" in data_path:
            srs.ImportFromProj4('+proj=geos +h=35785863 +a=6378137.0 +b=6356752.3 +lon_0=133.0 +no_defs')
        memDs.SetProjection(srs.ExportToWkt())
        memDs.SetGeoTransform([-5496000, int(res0), 0.0, 5496000, 0.0, -int(res0)])
        memDs.GetRasterBand(1).WriteArray(datavar)  # 写入数据

        # temp = memDs.ReadAsArray()
        # plt.imshow(temp)
        # plt.show()
        # plt.close()
        warpDs = gdal.Warp(dstFilePath, memDs, dstSRS=u'EPSG:4326', outputBounds=(llon, llat-res, ulon+res, ulat), xRes=res,
                           yRes=res, resampleAlg=gdalconst.GRA_Bilinear)  # 双线性插值, 也可是GRA_NearestNeighbour最近邻

        fy4_data = warpDs.ReadAsArray(0, 0, warpDs.RasterXSize, warpDs.RasterYSize)
        # plt.imshow(fy4_data)
        # plt.show()
        dataset = warpDs
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_bands = dataset.RasterCount  # 波段数
        im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
        im_proj = dataset.GetProjection()  # 获取投影信息
        x, y = np.meshgrid(np.arange(im_width), np.arange(im_height))
        lon = im_geotrans[0] + x * im_geotrans[1] + y * im_geotrans[2]
        lat = im_geotrans[3] + x * im_geotrans[4] + y * im_geotrans[5]

        del warpDs
        del memDs

        return 0, fy4_data,lon[0,:],lat[:,0]

    def FY4AL1PRO(self, mainfile, outputfile):
        all_chanel = []
        lon, lat = 0, 0
        for i in range(101, 115):

            C = str(i)[1:]
            # print(C, self.L1Channel[C])
            array = self.Read_FY4_Channel(mainfile, Channel=C)
            code, reproj, lon, lat = self.FY4_reproject(mainfile, array, llat=18.03125, ulat=54.03125, llon=73.03125, ulon=136.03125, res=0.0625)
            all_chanel.append(reproj)

        ds = xr.Dataset()
        ds.coords['lat'] = ('lat', lat)
        ds.coords['lon'] = ('lon', lon)

        ds['NOMChannel'] = (('chanel', 'lat', 'lon'), np.array(all_chanel))

        ds.to_netcdf(outputfile, encoding={'NOMChannel': {'zlib': True}})  # 对数据进行压缩

        return ds

    def Data_Cal(self,NOMChannel,CALChannel,sh):
        CALChannel=np.insert(CALChannel,0,0)
        NOMChannel[NOMChannel >sh] = 0
        NOMChannel = CALChannel[NOMChannel]
        return NOMChannel

    def re_datestr(self,namestr,data_format,index):
        mat = re.findall(data_format,namestr)
        mat_str = mat[index]
        mat_str = mat_str.replace('_','')
        return mat_str

    def run(self):
        time_coverage_start = self.re_datestr(self.primaryfile, r"(\d{14})", -2)
        # print(time_coverage_start)
        outputpath = os.path.join(self.outpath)
        if not os.path.exists(outputpath):
            os.makedirs(outputpath, mode=0o777)

        outfile = os.path.join(outputpath, "SATE_FY4A_AGRI_N_DISK_1047E_L1_GLL_"+time_coverage_start+"_4000M_Z_V001.NC") ############需按规范修改
        # if not os.path.exists(outfile):
        ds = self.FY4AL1PRO(self.primaryfile, outfile)
        return ds, outfile, time_coverage_start

def main(file_path, outrootpath):
    data = None
    if file_path.endswith('.hdf'):
        obj = Preprocess(primaryfile=file_path, outpath=outrootpath)
        data, outfile, time_coverage_start = obj.run()
        logging.info(f'数据处理完成！预处理数据被保存在：{outfile}。')
    return data


