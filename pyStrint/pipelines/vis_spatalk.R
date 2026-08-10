library(SpaTalk)
args <- commandArgs(trailingOnly = TRUE)
ligand = args[1]
receptor = args[2]
sender = args[3]
receiver = args[4]

sender_color = args[6]
receiver_color = args[7]
label_size = as.integer(args[8])
linewidth = as.numeric(args[9])
exp_threshold = as.numeric(args[10])
width = as.numeric(args[11])
height = as.numeric(args[12])
orientation = args[13]

# source('/data6/wangjingwan/5.Simpute/SpaTalk-main/R/plot.R', chdir = TRUE)
# source('/data6/wangjingwan/5.Simpute/SpaTalk-main/R/utils.R', chdir = TRUE)
script_path <- tryCatch(
  {
    # For non-interactive sessions
    new_args <- commandArgs(trailingOnly = FALSE)
    normalizePath(sub("--file=", "", new_args[grep("--file=", new_args)]))
  },
  error = function(e) {
    # For interactive sessions
    if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
      rstudioapi::getSourceEditorContext()$path
    } else {
      stop("Unable to determine script location.")
    }
  }
)

script_dir <- dirname(dirname(script_path))
script2_path <- file.path(script_dir, "/lib/SpaTalk/R/plot.R")
script3_path <- file.path(script_dir, "/lib/SpaTalk/R/utils.R")
source(script2_path, chdir = TRUE)
source(script3_path, chdir = TRUE)

setwd(args[5])
obj = readRDS('spatalk.rds')
# some name has /, which is not allowed in file name
sender_name <- gsub("/", "_", sender)
receiver_name <- gsub("/", "_", receiver)



# pdf(paste0("lrpair_",sender_name,'_',receiver_name,'_',ligand,'_',receptor,".pdf"),width=10,height=8)
# out = plot_lrpair(object = obj,ligand = ligand,receptor = receptor,
#                     celltype_sender = sender,
#                     celltype_receiver = receiver,            
#                     size = 4,
#                     arrow_length = 0.1)
# print(out)
# dev.off()

### 4.pathway
if (orientation == 'vertical') {
    vertical = TRUE
}else{
    vertical = FALSE
}
pdf(paste0("lr_path_",sender_name,'_',receiver_name,'_',ligand,'_',receptor,".pdf"),width=width,height=height)
out = plot_lr_path(object = obj,ligand = ligand,receptor = receptor,
                   celltype_sender = sender,
                   celltype_receiver = receiver,
                   color = c(sender_color, receiver_color),
                   linewidth = linewidth,
                   size = label_size,
                   exp_threshold = exp_threshold,
                   vertical = vertical)
print(out)
dev.off()
### 5.path2gene
# pdf(paste0("lr_path_gene_",sender_name,'_',receiver_name,'_',ligand,'_',receptor,".pdf"),width=8,height=8)
# out = plot_path2gene(object = obj,ligand = ligand,receptor = receptor,
#                      celltype_sender = sender,
#                      celltype_receiver = receiver)
# print(out)
# dev.off()
