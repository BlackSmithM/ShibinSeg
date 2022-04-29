
from osgeo import gdal, ogr, osr, gdalconst
import os
import datetime


def raster_to_shp(path, output_path):
    start_time = datetime.datetime.now()

    inraster = gdal.Open(path)  # 读取路径中的栅格数据
    inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息，用来为后面生成的矢量做准备

    outshp = os.path.join(output_path, path[:-4] + ".shp")  # 给后面生成的矢量准备一个输出文件名，这里就是把原栅格的文件名后缀名改成shp了
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  # 若文件已经存在，则删除它继续重新做一遍
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
    # 对shp文件创建一个图层，定义为多个面类
    Poly_layer = Polygon.CreateLayer(outshp[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)
    newField = ogr.FieldDefn('label', ogr.OFTInteger)  # 给目标shp文件添加一个字段，用来存储原始栅格的pixel value,浮点型，
    Poly_layer.CreateField(newField)

    gdal.Polygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作
    # gdal.FPolygonize(inband, None, Poly_layer, 0) # 只转矩形，不合并
    Polygon.SyncToDisk()
    Polygon = None
    end_time = datetime.datetime.now()
    print("Succeeded at", end_time)
    print("Elapsed Time:", end_time - start_time)  # 输出程序运行所需时间

def shp_to_raster(rasterFile, shpFile, output):
    # rasterFile = 'F:/**0416.dat'  # 原影像
    # shpFile = 'F:/**小麦.shp'  # 裁剪矩形

    dataset = gdal.Open('D:\DATA\SARDATA\S1A_IW_SLC__1SDV_20211029T103405_20211029T103433_040333_04C7A0_E3FB.SAFE\manifest.safe', gdalconst.GA_ReadOnly)

    geo_transform = dataset.GetGeoTransform()
    cols = dataset.RasterXSize  # 列数
    rows = dataset.RasterYSize  # 行数

    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * dataset.RasterXSize
    y_min = y_max + geo_transform[5] * dataset.RasterYSize

    pixel_width = geo_transform[1]

    shp = ogr.Open(shpFile, 0)
    m_layer = shp.GetLayerByIndex(0)
    target_ds = gdal.GetDriverByName('GTiff').Create(output, xsize=cols, ysize=rows, bands=1,
                                                     eType=gdal.GDT_Byte) # Int16
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -1 * pixel_width))
    # target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(dataset.GetProjection())

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], m_layer, options=["ATTRIBUTE=PC"])  # 跟shp字段给栅格像元赋值
    # gdal.RasterizeLayer(target_ds, [1], m_layer) # 多边形内像元值的全是255
    del dataset
    del target_ds
    shp.Release()



# shp_to_raster('4.bmp', r'D:\Desktop\Code\TianZhi\科目一样本标注示例1030\科目一样本标注_mask.shp', 'output/shp2tif.tif')
# raster_to_shp(r'C:\Users\BlackSmithM\Desktop\TianZhi\tools\output/shp2tif.tif', output_path='output')
shp_to_raster(r'D:\DATA\SARDATA\visible_data\21_cut_EnLee.jpg', r'D:\DATA\SARDATA\参考图斑\参考图斑.shp', r'D:\DATA\SARDATA\参考图斑\shp2tif.tif')
