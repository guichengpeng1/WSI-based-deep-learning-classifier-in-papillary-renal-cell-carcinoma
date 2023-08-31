from pathlib import Path
import os

kfb_path = r""#kfb.exe PATH 
kfb_source = r""# kfb file PATH

os.chdir(kfb_path)

from pathlib import Path
import os

kfb_path = r""
kfb_source = r""

os.chdir(kfb_path)



for kfb in Path(kfb_source).glob("*.kfb"):
    #print(str(kfb))
    #print(f'"{kfb}"')
    
    svs = Path(kfb).with_suffix(".svs")
    #svs = str(svs).replace(" ","")
    #kfb = str(kfb).replace(" ","?")
    #print(kfb)
   
    #print(f'"{svs}"')
    kfb = f'"{kfb}"'
    svs = f'"{svs}"'
    print(f"!KFbioConverter.exe {kfb} {svs} 9")

#eg:
!KFbioConverter.exe "*.kfb" "*.svs" 9
