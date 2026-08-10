
args <- commandArgs(trailingOnly = FALSE)
scriptPath <- normalizePath(sub("^--file=", "", args[grep("^--file=", args)]))
scriptPath <- dirname(scriptPath)

library(SpaTalk)

options(warn = 0)  

# load starmap data
args = commandArgs(trailingOnly = TRUE)
st_dir <- args[1]
st_meta_dir <- args[2]
sc_coord_dir <- args[3]
meta_key <- args[4]
species <- args[5]
out_f <- args[6]
out_dir = paste0(out_f,"/")
dir.create(file.path(out_dir), showWarnings = FALSE)

# print the arges
print(paste('st_dir:', st_dir))
print(paste('st_meta_dir:', st_meta_dir))
print(paste('sc_coord_dir:', sc_coord_dir))
print(paste('meta_key:', meta_key))
print(paste('species:', species))
print(paste('out_dir:', out_dir))

DEFAULT_N_CORES <- 4
if (length(args) > 6){
    n_cores = strtoi(args[7])
}else{
    n_cores = DEFAULT_N_CORES
}
print(paste('Using n_cores:', n_cores))


##########
# Helper function to read data with automatic delimiter detection
read_data <- function(file_path, transpose = FALSE) {
    if (grepl('\\.csv$', file_path)) {
        data <- read.table(file = file_path, sep = ',', header = TRUE, row.names = 1)
    } else {
        data <- read.table(file = file_path, sep = '\t', header = TRUE, row.names = 1)
    }
    
    if (transpose) {
        return(t(data))
    } else {
        return(data)
    }
}

# Load data files
st_data <- read_data(st_dir, transpose = TRUE)
st_meta <- read_data(st_meta_dir)
sc_coord <- read_data(sc_coord_dir)


if (species == 'Mouse'){
    if (grepl('cell_mapping_meta.tsv', sc_coord_dir)){
        sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
    }else if (grepl('before', sc_coord_dir)){
        sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
    }else if ('X' %in% colnames(sc_coord)){
        sc_coord = sc_coord[c('X','Y')]
    }else{
        sc_coord = sc_coord[c('x','y')]
    }
}

if (species == 'Human'){
    if (grepl('cell_mapping_meta.tsv', sc_coord_dir)){
        sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
    }else if (grepl('before', sc_coord_dir)){
        sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
    }else if (grepl('truth', sc_coord_dir)){
        sc_coord = sc_coord[c('X','Y')]
    }else{
        sc_coord = sc_coord[c('x','y')]
    }
}

print('loaded')

if (grepl('Human', species)){
    max_hop = 3
}else if ( 
    grepl('Mouse', species)){
    max_hop = 4
    # add neuronchat db
    lr_df_dir = paste0(scriptPath,"/../LR/mouse_LR_pairs.txt")
    new_lr_df = read.table(file = lr_df_dir, sep = '\t', header = FALSE)
    new_lr_df = new_lr_df[,c('V1','V2')]
    colnames(new_lr_df) = c('ligand','receptor')
    species = 'Mouse'
    new_lr_df$species = species
    new_lrpairs = rbind(lrpairs,new_lr_df)
    new_lrpairs$ligand = gsub("_", "-", new_lrpairs$ligand)
    new_lrpairs$receptor = gsub("_", "-", new_lrpairs$receptor)
    new_lrpairs = unique(new_lrpairs)
    lrpairs = new_lrpairs
}



# subset by meta index
st_data = st_data[,rownames(sc_coord)]
# Formating
sc_coord$cell = rownames(sc_coord)
sc_coord$cell <- sub("^", "C",sc_coord$cell)
sc_coord$cell = gsub("_", "-", sc_coord$cell)
colnames(sc_coord) = c('x','y','cell')
sc_coord = sc_coord[,c('cell','x','y')]

colnames(st_data) = sc_coord$cell
colnames(st_data) = gsub("_", "-", colnames(st_data))
rownames(st_data) = gsub("_", "-", rownames(st_data))
st_data = as.data.frame(st_data)

# print(head(sc_coord))
# print(head(st_data))

obj <- createSpaTalk(st_data = as.matrix(st_data),
                     st_meta = sc_coord,
                     species = species,
                     if_st_is_sc = T,
                     spot_max_cell = 1,celltype = st_meta[[meta_key]])
tp_lst = unique(obj@meta$rawmeta$celltype)


obj <- find_lr_path(object = obj , lrpairs = lrpairs, pathways = pathways, if_doParallel = T, use_n_cores=n_cores, max_hop = max_hop)
df = obj@lr_path$lrpairs
# print(df)
# df$chose = ifelse(grepl("true", df$receptor), "yes", "no")
# print(df[df$chose == "yes",])
for (tp1 in tp_lst) {
  for (tp2 in tp_lst) {
    if (tp1 != tp2) {
      tryCatch({
        obj <- dec_cci(object = obj, celltype_sender = tp1, celltype_receiver = tp2,
                       if_doParallel = T, use_n_cores = n_cores, pvalue = 0.1, n_neighbor = 20,
                       co_exp_ratio = 0.05, min_pairs = 2)
        obj <- dec_cci(object = obj, celltype_sender = tp2, celltype_receiver = tp1,
                       if_doParallel = T, use_n_cores = n_cores, pvalue = 0.1, n_neighbor = 20,
                       co_exp_ratio = 0.05, min_pairs = 2)
        print(tp1)
        print(tp2)
        # write.table(obj@lrpair, paste0(out_dir, "/lr_pair_append.csv"), row.names = TRUE, quote = FALSE, append = TRUE, sep = ",", col.names = FALSE)
      }, error = function(e) {
        cat("Error occurred during iteration: tp1:", tp1, "tp2:", tp2, "Error:", conditionMessage(e), "\n")
      })
      next
    }
  }
}

## obj <- dec_cci_all(object = obj, if_doParallel = T, use_n_cores=n_cores, pvalue=0.1, n_neighbor = 20, co_exp_ratio=0.05,min_pairs=2)
write.csv(obj@lrpair, paste0(out_dir,"/lr_pair.csv"), row.names = TRUE,quote = F)
saveRDS(obj, paste0(out_dir,"/spatalk.rds"))
############## LR ana ###################
# obj = readRDS(paste0(out_dir,"/spatalk.rds"))

r_object = obj@cellpair
df <- data.frame(
  Name = character(),
  cell_sender = character(),
  cell_receiver = character(),
  stringsAsFactors = FALSE
)

for (name in names(r_object)) {
  sender <- r_object[[name]]$cell_sender
  receiver <- r_object[[name]]$cell_receiver
  df <- rbind(df, data.frame(Name = rep(name, length(sender)), cell_sender = sender, cell_receiver = receiver, stringsAsFactors = FALSE))
}
write.csv(df, paste0(out_dir,"/cellpair.csv"), row.names = T,quote = F)
write.csv(obj@meta$rawmeta, paste0(out_dir,"/spatalk_meta.csv"), row.names = T,quote = F)