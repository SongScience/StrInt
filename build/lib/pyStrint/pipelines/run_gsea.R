# ref https://blog.csdn.net/qq_27390023/article/details/121318112
library(msigdb) # The Molecular Signatures Database (MSigDB)
library(fgsea)
# packageVersion("dplyr")
# packageVersion("vctrs")

# install.packages("dplyr")
# "hs" for human and "mm" for mouse
msigdb.hs = getMsigdb(org = 'hs',id = c("SYM", "EZID"))
# Downloading and integrating KEGG gene sets
msigdb.hs = appendKEGG(msigdb.hs)
length(msigdb.hs)

# data <- read.table('/data6/wangjingwan/5.Simpute/4.datasets/mela_c2l/spex/CAF_DEG.tsv', header = TRUE)
mydata <- read.table('/data6/wangjingwan/5.Simpute/4.datasets/mela_c2l/spex/CAF_DEG_fc.tsv', header = TRUE, row.names = 1)

# 可以根据需求选择子基因集
listCollections(msigdb.hs)

msigdb_ids <- geneIds(msigdb.hs)
FCgenelist = data['CAF_1_logfoldchanges']

mydata <- mydata[order(mydata$CAF_1_logfoldchanges,decreasing=TRUE),]
FCgenelist <- mydata$CAF_1_logfoldchanges
names(FCgenelist) <- mydata$CAF_1_names

fgseaRes <- fgsea(pathways = msigdb_ids, 
                  stats = FCgenelist,
                  minSize=15,
                  maxSize=500,
                  nperm=10000)
 
head(fgseaRes[order(pval), ])
sum(fgseaRes[, padj < 0.01])
fgseaRes_sub = fgseaRes[fgseaRes$pval < 0.01, ]
fgseaRes_sub = fgseaRes_sub[order(NES,pval), ]
up_paths = fgseaRes_sub[ES > 0][head(order(pval), n=10),]

fgseaRes_df = as.data.frame(fgseaRes_sub[order(pval), c('pathway','pval','padj','ES','NES','size') ])
write.table(fgseaRes_df, file = "fgseaRes.tsv", sep = "\t", quote = FALSE)

topPathwaysUp <- fgseaRes_sub[ES > 0][head(order(pval), n=10), pathway]
topPathwaysDown <- fgseaRes_sub[ES < 0][head(order(pval), n=10), pathway]
topPathways <- c(topPathwaysUp, rev(topPathwaysDown))
 
# 画table图
tiff('enriched_pathway.tiff', units="in", width=8, height=6, res=600, compression = 'lzw')
plotGseaTable(msigdb_ids[topPathways], FCgenelist, fgseaRes_sub, 
               gseaParam=0.5)
dev.off()
# 画通基因集的富集图， HALLMARK_HYPOXIA 为一种基因集名称。
plotEnrichment(msigdb_ids[["HALLMARK_HYPOXIA"]],FCgenelist)+ 
  labs(title="HALLMARK_HYPOXIA")