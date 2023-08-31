library(rms)
KIRP <- read.csv(file="E:/R workplace/TCGA.csv",header = TRUE)
dd<-datadist(KIRP)
options(datadist="dd")
KIRP$cluster.of.clusters..CoCA.
####计算c-index:Multimodal.score3 pRCC_dpscore Risk.rank deep.learning.classifier  ##
####PFS.time,PFS.status
f<-cph(Surv(PFS.time,PFS.status)~risk.score
       ,data=KIRP,x=TRUE,y=TRUE,surv = TRUE)
rcorrcensresult<-rcorrcens(Surv(PFS.time,PFS.status) ~ predict(f), data = KIRP)
(rcorrcensresult[1,3])/2+0.5   ##C指数计算

(rcorrcensresult[1,3])/2+0.5-rcorrcensresult[1,4]*1.96  ###95%CI low
(rcorrcensresult[1,3])/2+0.5+rcorrcensresult[1,4]*1.96  ###95%CI high