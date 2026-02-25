
# Rで処理するためのスクリプト

########## readme
# 0. 準備体操
# 1. エクセルから実験データの読み出し(2バイト文字は使えないので半角に変換)
# 2. 未時効材など計算しないデータを除外
# 3. Thermo-Calcデータを読み出して格納する
# 4. 実験データと計算データの差を計算して、誤差の絶対値の和を出力

################# 
# 0. 
# install.packages("xlsx", dep=T);
# JAVA_HOMEが必要

# library(xlsx);
# gphaseEXP <- read.xlsx("20210128_hus_data.xlsx", sheetName="GPHASE_data")
# saveRDS(gphaseEXP, file = "20210128_hus_data_gphase.obj")


library(sigmoid)   # to use LeRU func.

################# 
# 1.
# Rで処理する関係上、2バイト文字は使わない
# x <- read.xlsx("C:/temp/mydata.xlsx", sheetName="Sheet1")

# library(xlsx);
# gphaseEXP <- read.xlsx("20210501_hus_data.xlsx", sheetName="GPHASE_data")
# plot(gphaseEXP)

# saveRDS(gphaseEXP, file = "20210501_hus_data_gphase.obj")
# gphaseEXP   <-  readRDS("20210501_hus_data_gphase.obj")
# gphaseEXP   <-  gphaseEXP[-64,]  

gphaseEXP  <-  read.csv("exp_data_for_dakota_opti.csv",header=T)


################# 
# 2.
# numdata <- c(1, 61)
# numdata <- c(1, 4, 22, 61) 
# 全データ利用(未時効データ除く)
# numdata <-  c(1:63)
# gphaseEXP  <-   gphaseEXP[c(-32, seq(-57,-46)),]
# numdata <-  numdata[c(-32, seq(-57,-46))]


# 実験データを6点で最適化する
# EXPdat    <- c(1, 38, 42, 61, 64, 65)
#EXPdat    <- c(5, 8, 21, 45, 66, 67)
# EXPdat    <- c(1,2,4,  7,11,16,   28,31,38,40,41,42,43,45,60,61,62,63,64,65)
EXPdat      <- c(1,2,4,5,7,11,16,21,      38,40,      43,45,60,61,62,63,      66,67)
numEXP    <-  formatC(EXPdat, width = 2, flag = "0")
#numEXP   <- c("01", "38", "42", "61", "64", "65")
# データの抽出
gphaseEXP <- gphaseEXP[EXPdat,]
filenames   <-  paste(numEXP, "S.PLE", sep="")



################# 
# 3.
# Thermo-Calcの結果ファイルPLEの解析

# cat *S.PLE | grep G_PHASE -A 4 | grep Moles | awk '{print $2}' | sed s/,//g
cmdMoles   <-   paste("cat ", filenames, " | grep G_PHASE -A 4 | grep Moles | awk '{print $2}' | sed s/,//g")
# cat *S.PLE | grep G_PHASE -A 4 | grep force | awk '{print $6}' 
cmdDF   <-   paste("cat ", filenames, " | grep G_PHASE -A 4 | grep force | awk '{print $6}'")


# for Si amount
# デミリタを二文字にできないので、いったんQに変換
# cat *S.PLE | grep G_PHASE -A 4 | grep FE  | sed s/FE/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3
# cat *S.PLE | grep G_PHASE -A 4 | grep NI  | sed s/NI/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3
# cat *S.PLE | grep G_PHASE -A 4 | grep SI  | sed s/SI/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3
# cat *S.PLE | grep G_PHASE -A 4 | grep MN  | sed s/MN/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3
# cat *S.PLE | grep G_PHASE -A 4 | grep MO  | sed s/MO/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3
# cat *S.PLE | grep G_PHASE -A 4 | grep CR  | sed s/CR/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3
# cat *S.PLE | grep G_PHASE -A 4 | grep C   | sed s/' C '/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3
# cat *S.PLE | grep G_PHASE -A 4 | grep N   | sed s/' N '/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3


cmdFE   <-   paste("cat ", filenames, "| grep G_PHASE -A 4 | grep FE  | sed s/FE/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3")
cmdNI   <-   paste("cat ", filenames, "| grep G_PHASE -A 4 | grep NI  | sed s/NI/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3")
cmdSI   <-   paste("cat ", filenames, "| grep G_PHASE -A 4 | grep SI  | sed s/SI/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3")
cmdMO   <-   paste("cat ", filenames, "| grep G_PHASE -A 4 | grep MO  | sed s/MO/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3")
cmdMN   <-   paste("cat ", filenames, "| grep G_PHASE -A 4 | grep MN  | sed s/MN/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3")
cmdCR   <-   paste("cat ", filenames, "| grep G_PHASE -A 4 | grep CR  | sed s/CR/Q/g | grep Q | cut -d 'Q' -f 2 | cut -d ' ' -f 3")


error1exp  <-   c()
visdata    <-   c()
par(mfrow=c(3,2))
caldata   <-   c(0,0,0,0,0,0)  # Fe, Ni, Si, Mn, Mo, Cr

for (i in c(1:length(numEXP))){
    moles   <-   system(cmdMoles[i], intern=T)   # for Moles
    moles_tot   <-   sum(as.numeric(moles))
    dfs     <-   system(cmdDF[i], intern=T)      # for DF
    nmoles  <-   which(as.numeric(moles)!=0.0)
# G相が出ていない場合
    if (moles_tot == 0.0){
        caldata  <-   c(0,0,0,0,0,0)
        }else if (length(nmoles) == 1){
# G相が一つしか出ていない場合 
        caldata  <-   c(sum(as.numeric(system(cmdFE[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdNI[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdSI[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdMN[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdMO[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdCR[i],intern = TRUE)[nmoles])))
# G相が分離している場合    
        }else if (length(nmoles) >= 2){  
        caldata  <-   c(sum(as.numeric(system(cmdFE[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdNI[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdSI[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdMN[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdMO[i],intern = TRUE)[nmoles])),
                        sum(as.numeric(system(cmdCR[i],intern = TRUE)[nmoles])))
        caldata   <-   10*caldata
        }

    expdata  <-  c(gphaseEXP[i,1], gphaseEXP[i,2], gphaseEXP[i,3], gphaseEXP[i,4], gphaseEXP[i,5], gphaseEXP[i,6])
# expdataの規格化(規格化が不要な場合は下の二行をコメントアウトする)
    coeffexp  <-  1/sum(expdata)
    expdata <- expdata*coeffexp


    visdata   <-  rbind(visdata, expdata)
    visdata   <-  rbind(visdata, caldata)

    diffcomp0     <-  expdata - caldata
    barplot(diffcomp0, names.arg=c("Fe","Ni","Si","Mn","Mo","Cr"), ylim=c(-0.4, 0.4), main=filenames[i]); grid(); abline(h=0 , col=2)
     diffcomp0[1]   <-   0.25*diffcomp0[1]   # for Fe
     diffcomp0[2]   <-   1*diffcomp0[2]   # for Ni
     diffcomp0[3]   <-   1*diffcomp0[3]   # for Si
     diffcomp0[4]   <-   1*diffcomp0[4]   # for Mn 
     diffcomp0[5]   <-   1*diffcomp0[5]   # for Mo 
     diffcomp0[6]   <-   0.25*diffcomp0[6]   # for Cr 
#    diffcomp       <-  expdata / caldata
    error1exp[i]  <-  sum(abs(diffcomp0))
    }


error_total  <-  sum(error1exp)/length(numEXP)
# error_total  <-  (sum(error_tot) + 1*sum(error_Temp))/length(numEXP)
# error_total  <-  sum(error_tot[numdagT])
write(error_total, file="error_tot.txt") 


### plot each data
par(mfrow=c(1,1))
par(mar=c(5,5,5,8))
par(xpd=T)   # 枠外への描画を許可
# 凡例を表示
# par()$usr[2], par()$usr[4]で右上の座標を取得する
# legend(par()$usr[2], par()$usr[4], legend="Elem.")


color=c("lightblue", "lightcyan", "lavender", "mistyrose", "cornsilk", "lightgreen")

colnames(visdata) <- c("Fe", "Ni", "Si", "Mn", "Mo", "Cr")
# rownames(visdata) <- rev(numEXP)
labexp  <-  paste(numEXP,"Exp")
labcal  <-  paste(numEXP,"Cal")
rownames(visdata)  <-  c(rbind(labexp, labcal))
# barplot(t(visdata), legend = TRUE, horiz=T)
visdata2  <-  t(apply(visdata,2,rev))
bp <- barplot(visdata2, main = "Comp. in G-phase (Exp./Calc.)", legend = F, horiz=T, col=color, cex.axis=0.8,  cex.names=0.6, las=1)
legend(x=par()$usr[2],y=par()$usr[4],legend=rownames(visdata2),pch=15,lty=0,xjust=-.1,col=color, pt.cex=2,x.intersp=2,y.intersp=1.5)#, title="Exp.")
for (j in 1:ncol(visdata2)) {
        xl <- bp[j]
        yl <- cumsum(visdata2[,j]) - visdata2[,j]/2
        txt <- 1.00*round(visdata2[,j], 3)
        txt[txt=="0"] <- NA; text(yl, xl, txt,cex=0.5)
        }

