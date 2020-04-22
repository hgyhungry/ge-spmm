#!/bin/bash
cd snap/

wget https://sparse.tamu.edu/MM/SNAP/soc-Epinions1.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/soc-LiveJournal1.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/soc-Slashdot0811.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/soc-Slashdot0902.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/wiki-Vote.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/email-EuAll.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/email-Enron.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/wiki-Talk.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/cit-HepPh.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/cit-HepTh.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/cit-Patents.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/ca-AstroPh.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/ca-CondMat.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/ca-GrQc.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/ca-HepPh.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/ca-HepTh.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/web-BerkStan.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/web-Google.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/web-NotreDame.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/web-Stanford.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/amazon0302.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/amazon0312.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/amazon0505.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/amazon0601.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/p2p-Gnutella04.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/p2p-Gnutella05.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/p2p-Gnutella06.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/p2p-Gnutella08.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/p2p-Gnutella09.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/p2p-Gnutella24.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/p2p-Gnutella25.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/p2p-Gnutella30.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/p2p-Gnutella31.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/roadNet-CA.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/roadNet-PA.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/roadNet-TX.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/as-735.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/as-Skitter.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/as-caida.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/Oregon-1.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/Oregon-2.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/soc-sign-epinions.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/soc-sign-Slashdot081106.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/soc-sign-Slashdot090216.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/soc-sign-Slashdot090221.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/CollegeMsg.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/com-Amazon.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/com-DBLP.tar.gz
# wget https://sparse.tamu.edu/MM/SNAP/com-Friendster.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/com-LiveJournal.tar.gz
# wget https://sparse.tamu.edu/MM/SNAP/com-Orkut.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/com-Youtube.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/email-Eu-core.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/email-Eu-core-temporal.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/higgs-twitter.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/loc-Brightkite.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/loc-Gowalla.tar.gz
# wget https://sparse.tamu.edu/MM/SNAP/soc-Pokec.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/soc-sign-bitcoin-alpha.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/soc-sign-bitcoin-otc.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/sx-askubuntu.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/sx-mathoverflow.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/sx-stackoverflow.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/sx-superuser.tar.gz
# wget https://sparse.tamu.edu/MM/SNAP/twitter7.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/wiki-RfA.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/wiki-talk-temporal.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/wiki-topcats.tar.gz


find . -name '*.tar.gz' -exec tar xvf {} \;
rm *.tar.gz
cp ../conv.c .
gcc -O3 -o conv conv.c

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
mv ${ii}.mtx ${ii}.mt0
../conv ${ii}.mt0 ${ii}.mtx 
rm ${ii}.mt0
cd ..
done

rm conv conv.c