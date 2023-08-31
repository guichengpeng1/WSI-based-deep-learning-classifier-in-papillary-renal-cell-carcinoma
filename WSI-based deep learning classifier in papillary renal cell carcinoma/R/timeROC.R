######: https://github.com/guichengpeng1
######:Chengpeng.Gui@uhn.ca

#install.packages("survival")
#install.packages("survminer")
#install.packages("timeROC")



library(survival)
library(survminer)
library(timeROC)
riskFile="risk.txt"       
cliFile="clinical.txt"     
setwd("D:\\AI\\ROC")    


risk=read.table(riskFile, header=T, sep="\t", check.names=F, row.names=1)
risk=risk[,c("futime", "fustat", "riskScore", "riskScore1")]


cli=read.table(cliFile, header=T, sep="\t", check.names=F, row.names=1)


samSample=intersect(row.names(risk), row.names(cli))
risk1=risk[samSample,,drop=F]
cli=cli[samSample,,drop=F]
rt=cbind(risk1, cli)


bioCol=rainbow(ncol(rt)-1, s=0.9, v=0.9)


ROC_rt=timeROC(T=risk$futime,delta=risk$fustat,
	           marker=risk$riskScore,cause=1,
	           weighting='marginal',iid=TRUE,
	           times=c(3,5,7),ROC=TRUE)

pdf(file="ROC.pdf", width=5, height=5)
plot(ROC_rt,time=3,col=bioCol[1],title=FALSE,lwd=2)
plot(ROC_rt,time=5,col=bioCol[2],add=TRUE,title=FALSE,lwd=2)
plot(ROC_rt,time=7,col=bioCol[3],add=TRUE,title=FALSE,lwd=2)
legend('bottomright',
       c(paste0('AUC at 3 years: ',sprintf("%.03f",ROC_rt$AUC[1])),
         paste0('AUC at 5 years: ',sprintf("%.03f",ROC_rt$AUC[2])),
         paste0('AUC at 7 years: ',sprintf("%.03f",ROC_rt$AUC[3]))),
       col=bioCol[1:3], lwd=2, bty = 'n')
dev.off()




ROC_rt1=timeROC(T=risk$futime,delta=risk$fustat,
               marker=risk$riskScore1,cause=1,
               weighting='marginal',iid=TRUE,
               times=c(3,5,7),ROC=TRUE)
confint(ROC_rt,level = 0.9,parm=NULL,n.sim=2000)
#confint(ROC_rt1,level = 0.95)
#confint(ROC_rt,level = 0.95)
compare(ROC_rt,ROC_rt1)


#aalen
######: https://github.com/guichengpeng1
######:Chengpeng.Gui@uhn.ca
