import os
import gdal, gdalconst, ogr

def pol2line(polyfn, linefn):
    """
        This function is used to make polygon convert to line
    :param polyfn: the path of input, the shapefile of polygon
    :param linefn: the path of output, the shapefile of line
    :return:
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    polyds = ogr.Open(polyfn, 0)
    polyLayer = polyds.GetLayer()
    spatialref = polyLayer.GetSpatialRef()
    #创建输出文件
    if os.path.exists(linefn):
        driver.DeleteDataSource(linefn)
    lineds =driver.CreateDataSource(linefn)
    linelayer = lineds.CreateLayer(linefn, srs=spatialref, geom_type=ogr.wkbLineString)
    featuredefn = linelayer.GetLayerDefn()
    #获取ring到几何体
    #geomline = ogr.Geometry(ogr.wkbGeometryCollection)
    for feat in polyLayer:
        geom = feat.GetGeometryRef()
        ring = geom.GetGeometryRef(0)
        #geomcoll.AddGeometry(ring)
        outfeature = ogr.Feature(featuredefn)
        outfeature.SetGeometry(ring)
        linelayer.CreateFeature(outfeature)
        outfeature = None


if __name__ == "__main__":
    pol2line(r'C:\Users\BlackSmithM\Desktop\TianZhi\科目一样本标注示例1030\科目一样本标注_mask.shp', 'output/line_result.shp')