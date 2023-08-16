#/GP1
cd gozu-tab/mydataset
for file in gp5_files/2/*.gp5
do
echo ${file}
echo ${file:12:-4}
python test.py encode ${file} labels/2/${file:12:-4}.json unknown
done