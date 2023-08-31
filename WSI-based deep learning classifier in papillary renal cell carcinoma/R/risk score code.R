######: https://github.com/guichengpeng1
######:Chengpeng.Gui@uhn.ca

setwd("J:/dab2ip data/20150601")
TCGA.risk.score <- read.csv("J:/dab2ip data/20150601/TCGA risk score.csv")

col=ifelse(TCGA.risk.score$status>0,"red","green")
plot(TCGA.risk.score$cg14122599,TCGA.risk.score$time,col=col)

abline(v=50,lwd=1,col="blue")


setwd("//Volumes//AI//TRAIN")
TCGA.risk.score <- read.csv("//Volumes//train.csv")
#risk score
data=TCGA.risk.score
data=data.matrix(TCGA.risk.score)
data=data.frame(data)

data=data[order(data$"leibovichscore711",decreasing=TRUE),]
height=data$"leibovichscore711"
height1=height


color=ifelse(data$"status"==0,"gray80","#E18727")
win.graph(width=21, height=10,pointsize=8)
barplot(height1, width = 1, space = 0,
        names.arg = NULL, legend.text = NULL, beside = FALSE,
        horiz = F, density = NULL, angle = 45,
        col = color, border=FALSE,
        main = NULL, sub = NULL, xlab = NULL, ylab = NULL,
        xlim = NULL, ylim = c(0,10), xpd = TRUE, log = "",
        axes = TRUE, axisnames = TRUE,
        cex.axis = par("cex.axis"), cex.names = par("cex.axis"),
        plot = TRUE, axis.lty = 0, offset = 0,
        add = FALSE, args.legend = NULL)





#6E819F","#D48286
#6f99adff","#EE4c97ff"
#6f99adff","#1b1919ff
#gray80","#1b1919ff"

######: https://github.com/guichengpeng1
######:Chengpeng.Gui@uhn.ca
