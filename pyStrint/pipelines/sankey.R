args <- commandArgs(trailingOnly = TRUE)
library(ggalluvial)
library(ggpubr)
# setwd('/data6/wangjingwan/5.Simpute/4.datasets/BR_c2l_S1_rep50//spade/')
# fn = 'pattern2_lr_flow.tsv'
# levels = 'Before,After'
# colors = '#EC7063,#D95F02,#ffb200,#D2B4DE,#E88AC2'
fn = args[1]
levels = args[2]
colors = args[3]
setwd(args[4])
levels = strsplit(levels, ',')[[1]]
colors = strsplit(colors, ',')[[1]]
data = read.table(file = fn, sep = '\t', header = TRUE,row.names = 1)
colors = colors
data$source<-factor(data$source,levels=levels)
data$Celltype<-factor(data$Celltype)

out = paste0("tp_flow.pdf")
pdf(file=out, width=8, height=5)
ggplot(data=data,aes(x=source,y=Propotion,alluvium = Celltype,stratum=Celltype)) + 
  geom_alluvium(aes(fill=Celltype,color=Celltype),alpha = .3,width = 0.7) + 
  geom_stratum(aes(fill=Celltype,color=Celltype),width = 0.7,alpha=.8) +
  scale_color_manual(values= colors) + scale_fill_manual(values=colors) + theme_classic(base_size = 26)
dev.off()

