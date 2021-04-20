from os import path, listdir
import os
import cv2


if __name__ == '__main__':
    desktop = os.path.expanduser("~/catkin_workspace")
    src = desktop+"/src/Handle_detection/handle400/data_v0/labels/test_1"
    des = desktop+"/src/Handle_detection/handle400/data_v0/labels/test"

    for f in listdir(src):
    	with open(path.join(src,f),'r') as infile:
            f_new = f.split(".")[0] + ".txt"
            with open(path.join(des,f_new), "w") as output: 
                lines = infile.readlines()
                for line_number, line in enumerate(lines):
                    #print("true or false:::"+str(line.split("<name>")[0] == "		"))
                    #if (line.split("<name>")[0] == "		"):
                    #    print(line.split("<name>")[1])
                    #if (line.split("<width>")[0]=="		"):
                    #    if (line.split("<width>")[1].split("</width>")[0] != "900"):
                    #        print("false")
                    if (line.split("<name>")[0] == "		") and (line.split("<name>")[1] == "doorHandle</name>\n"):
                            x_min = float(lines[line_number+5].split("<xmin>")[1].split("</xmin>")[0])
                            y_min = float(lines[line_number+6].split("<ymin>")[1].split("</ymin>")[0])
                            x_max = float(lines[line_number+7].split("<xmax>")[1].split("</xmax>")[0])
                            y_max = float(lines[line_number+8].split("<ymax>")[1].split("</ymax>")[0])
                            x_center = (x_max + x_min)/(2*900)
                            y_center = (y_max + y_min)/(2*1200)
                            x_range = (x_max - x_min)/900
                            y_range = (y_max - y_min)/1200
                            output.write("0 "+str(x_center)+" "+str(y_center)+" "+str(x_range)+ " "+str(y_range)+"\n")
