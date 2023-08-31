######: https://github.com/guichengpeng1
######:Chengpeng.Gui@uhn.ca

#install.packages("survival")
#install.packages("survminer")



library(survival)
library(survminer)
setwd("//Volumes//AI//cohort")     

bioSurvival=function(inputFile=null,outFile=null){
	
	rt=read.table(inputFile,header=T,sep="\t",check.names=F)
	
	diff=survdiff(Surv(futime, fustat) ~risk,data = rt)
	pValue=1-pchisq(diff$chisq,df=1)
	if(pValue<0.001){
		pValue="p<0.001"
	}else{
		pValue=paste0("p=",sprintf("%.03f",pValue))
	}
	fit <- survfit(Surv(futime, fustat) ~ risk, data = rt)
		
	
	surPlot=ggsurvplot(fit, 
		           data=rt,
		           conf.int=F,
		           pval=pValue,
		           pval.size=6,
		           legend.title="Risk",
		           legend.labs=c("H", "L"),
		           xlab="Time(years)",
		           break.time.by = 3,
		           palette=c("#BD3C29","#0172B6"),
		           risk.table='abs_pct',
		           risk.table.title="",
		           ncensor.plot=F,
		           censor.shape=124,censor.size=2.5,
		           risk.table.height=.30)
	pdf(file=outFile,onefile = FALSE,width = 7,height =5.5)
	print(surPlot)
	dev.off()
}
bioSurvival(inputFile="trainRisk.txt", outFile="OS.pdf")


##"#FFD500FF","00E5FFFF"  ,"#0072b5"
######: https://github.com/guichengpeng1
######:Chengpeng.Gui@uhn.ca
