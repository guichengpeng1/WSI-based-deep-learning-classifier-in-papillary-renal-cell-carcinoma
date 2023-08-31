######: https://github.com/guichengpeng1
######:Chengpeng.Gui@uhn.ca
#install.packages("survival")
#install.packages("regplot")



library(survival)
library(regplot)
riskFile="risk.txt"       
cliFile="clinical.txt"   
setwd("D:\\Nomo")     

risk=read.table(riskFile, header=T, sep="\t", check.names=F, row.names=1)
risk=risk[,c("futime", "fustat", "risk")]


cli=read.table(cliFile, header=T, sep="\t", check.names=F, row.names=1)
cli=cli[apply(cli,1,function(x)any(is.na(match('unknow',x)))),,drop=F]
cli$Age=as.numeric(cli$Age)


samSample=intersect(row.names(risk), row.names(cli))
risk1=risk[samSample,,drop=F]
cli=cli[samSample,,drop=F]
rt=cbind(risk1, cli)


res.cox=coxph(Surv(futime, fustat) ~ . , data = rt)
nom1<-regplot(res.cox,
              plots = c("density", "boxes"),
              clickable=F,
              title="",
              points=TRUE,
              droplines=TRUE,
              observation=rt[1,],
              rank="sd",
              failtime = c(1,3,5),
              prfail = T)


######: https://github.com/guichengpeng1
######:Chengpeng.Gui@uhn.ca
