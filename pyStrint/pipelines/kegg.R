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
    gene_ids <- as.character(gene$ENTREZID)
    gene_ids <- unique(gene_ids[!is.na(gene_ids) & gene_ids != ""])

    term2gene_raw <- read.delim(paste0("https://rest.kegg.jp/link/pathway/", org), header = FALSE, stringsAsFactors = FALSE)
    term2gene <- data.frame(
      ID = sub("^path:", "", term2gene_raw$V2),
      ENTREZID = sub(paste0("^", org, ":"), "", term2gene_raw$V1),
      stringsAsFactors = FALSE
    )
    term2name_raw <- read.delim(paste0("https://rest.kegg.jp/list/pathway/", org), header = FALSE, stringsAsFactors = FALSE)
    term2name <- data.frame(
      ID = sub("^path:", "", term2name_raw$V1),
      Description = sub(" - .*", "", term2name_raw$V2),
      stringsAsFactors = FALSE
    )

    kk <- enricher(
      gene = gene_ids,
      TERM2GENE = term2gene,
      TERM2NAME = term2name,
      pvalueCutoff = 0.05,
      pAdjustMethod = "BH",
      qvalueCutoff = 0.2
    )

    if (!is.null(kk) && nrow(as.data.frame(kk)) > 0) {
      res <- as.data.frame(kk)
      res$category <- NA_character_
      res$subcategory <- NA_character_
      cols <- c("category", "subcategory", "ID", "Description", "GeneRatio", "BgRatio",
                "RichFactor", "FoldEnrichment", "zScore", "pvalue", "p.adjust", "qvalue",
                "geneID", "Count")
      keep_cols <- cols[cols %in% colnames(res)]
      res <- res[, keep_cols, drop = FALSE]
      write.table(res, file = new_file_name, sep = '\t', quote = FALSE, row.names = FALSE)
      pdf(file = paste0(new_file_name, ".pdf"), width = 8, height = 10)
      dotplot(kk, showCategory = 30)
      dev.off()
    } else {
      empty_res <- data.frame(
        category = character(),
        subcategory = character(),
        ID = character(),
        Description = character(),
        GeneRatio = character(),
        BgRatio = character(),
        RichFactor = numeric(),
        FoldEnrichment = numeric(),
        zScore = numeric(),
        pvalue = numeric(),
        p.adjust = numeric(),
        qvalue = numeric(),
        geneID = character(),
        Count = integer(),
        stringsAsFactors = FALSE
      )
      write.table(empty_res, file = new_file_name, sep = '\t', quote = FALSE, row.names = FALSE)
    }
  }, error = function(e) {
    # Handle the error here or print an error message
    print(paste("Error occurred for file:", fn))
  })
}
