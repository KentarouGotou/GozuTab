#/GP1
cd gozu-tab/mydataset
for file in labels/2/*.json
do
echo ${file:9:-5}
python convert.py encode ${file} clean0 converted/${file:9}
done