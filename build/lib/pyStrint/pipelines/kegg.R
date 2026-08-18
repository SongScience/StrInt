# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("clusterProfiler")
# BiocManager::install(c("org.Hs.eg.db", "org.Mm.eg.db"))
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(clusterProfiler))

setwd(args[1])
if (args[2] == 'Human') {
  suppressPackageStartupMessages(library(org.Hs.eg.db))
  ref_db <- org.Hs.eg.db
  org = 'hsa'
} else if (args[2] == 'Mouse') {
  suppressPackageStartupMessages(library(org.Mm.eg.db))
  ref_db <- org.Mm.eg.db
  org = 'mmu'
}


if (length(args) > 2) {
  fn_lst = c(args[3])
}else{
  fn_lst = list.files('./', pattern = "kegg.tsv")
}


for (fn in fn_lst) {
  print(fn)
  tryCatch({
    top <- read.table(file = fn, sep = '\t', header = TRUE, row.names = 1)
    new_file_name <- paste0(sub("\\.tsv$", "", fn), "_enrichment", ".tsv")
    gene <- bitr(top$gene, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = ref_db)
    gene$logFC <- top$lr_co_exp_num[match(gene$SYMBOL, top$gene)]
    write.table(gene, file = paste0(sub("\\.tsv$", "", fn), "_geneID", ".tsv"), sep = '\t', quote = FALSE)
    geneList <- gene$logFC
    names(geneList) <- gene$ENTREZID
    geneList <- sort(geneList, decreasing = TRUE)
    head(geneList)
    gene <- names(geneList)[abs(geneList) > 0]
    
    kk <- enrichKEGG(gene = gene, organism = org, pvalueCutoff = 0.05)
    
    write.table(kk@result, file = new_file_name, sep = '\t', quote = FALSE)
    pdf(file = paste0(new_file_name, ".pdf"), width = 8, height = 10)
    dotplot(kk, showCategory = 30)
    dev.off()
  }, error = function(e) {
    # Handle the error here or print an error message
    print(paste("Error occurred for file:", fn))
  })
}