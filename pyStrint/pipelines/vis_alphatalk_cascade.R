args <- commandArgs(trailingOnly = TRUE)
in_tsv <- args[1]
out_dir <- args[2]
ligand <- args[3]
receptor <- args[4]
sender <- args[5]
receiver <- args[6]
sender_color <- args[7]
receiver_color <- args[8]
label_size <- as.numeric(args[9])
linewidth <- as.numeric(args[10])
w <- as.numeric(args[11])
h <- as.numeric(args[12])
max_hop <- as.integer(args[13])
max_nodes_per_hop <- as.integer(args[14])
max_yes_per_hop <- 8
max_no_per_hop <- 8

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(dplyr)
})

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

x <- read.delim(in_tsv, sep='\t', header=TRUE, stringsAsFactors = FALSE)
x <- x %>% filter(sender_major == sender, receiver_major == receiver)
if (nrow(x) == 0) stop('No AlphaTalk rows for selected sender/receiver.')

# download/load SpaTalk pathways table
pathways_rda <- file.path(out_dir, 'pathways.rda')
if (!file.exists(pathways_rda)) {
  download.file('https://raw.githubusercontent.com/ZJUFanLab/SpaTalk/main/data/pathways.rda', pathways_rda, mode='wb', quiet=TRUE)
}
load(pathways_rda)  # object: pathways
if (!exists('pathways')) stop('Failed to load pathways.rda')

pathways <- pathways[pathways$species %in% c('Human', 'Mouse'), ]
# infer species from gene symbol style if possible; default Human
sp <- 'Human'
if ('species' %in% colnames(x) && length(unique(x$species)) > 0) {
  sp <- unique(x$species)[1]
}
pathways <- pathways[pathways$species == sp, ]

# expression-supported gene universe from current FB->T alpha rows
gene_set <- unique(c(x$ligand, x$receptor))
pathways <- pathways[pathways$src %in% gene_set | pathways$dest %in% gene_set | pathways$src == receptor, ]

# BFS cascade from receptor on directed graph src->dest
frontier <- c(receptor)
seen <- c(receptor)
node_hop <- data.frame(node = receptor, hop = 2, tf='NO', stringsAsFactors = FALSE)
edge_df <- data.frame(src = ligand, dst = receptor, stringsAsFactors = FALSE)

for (hop in 3:(max_hop + 2)) {
  step <- pathways[pathways$src %in% frontier, c('src','dest','dest_tf')]
  if (nrow(step) == 0) break
  step <- step[!duplicated(step[, c('src','dest')]), ]
  step$dest_tf <- ifelse(step$dest_tf == 'YES', 'YES', 'NO')
  if ('co_exp_number' %in% colnames(x)) {
    step <- merge(step, unique(x[, c('ligand', 'receptor', 'co_exp_number')]),
                  by.x = c('src', 'dest'), by.y = c('ligand', 'receptor'), all.x = TRUE)
    step$co_exp_number[is.na(step$co_exp_number)] <- -1
    step <- step[order(-step$co_exp_number, step$src, step$dest), ]
  } else {
    step <- step[order(step$src, step$dest), ]
  }
  # Keep a balanced set per hop: up to 8 TF(YES) and up to 8 non-TF(NO).
  step_yes <- step[step$dest_tf == 'YES', ]
  step_no <- step[step$dest_tf != 'YES', ]
  step_yes <- head(step_yes, min(max_yes_per_hop, max_nodes_per_hop))
  remain <- max_nodes_per_hop - nrow(step_yes)
  step_no <- head(step_no, min(max_no_per_hop, max(0, remain)))
  step <- rbind(step_yes, step_no)
  edge_df <- rbind(edge_df, data.frame(src = step$src, dst = step$dest, stringsAsFactors = FALSE))

  new_nodes <- unique(step$dest[!step$dest %in% seen])
  if (length(new_nodes) == 0) break

  tf_map <- step %>% group_by(dest) %>% summarise(tf = ifelse(any(dest_tf == 'YES'),'YES','NO'), .groups='drop')
  node_hop <- rbind(node_hop, data.frame(node = new_nodes, hop = hop, tf = tf_map$tf[match(new_nodes, tf_map$dest)], stringsAsFactors = FALSE))
  seen <- c(seen, new_nodes)
  frontier <- new_nodes
}

# keep only connected nodes
nodes <- unique(c(edge_df$src, edge_df$dst))
node_hop <- node_hop[node_hop$node %in% nodes, ]
if (!ligand %in% node_hop$node) {
  node_hop <- rbind(data.frame(node=ligand, hop=1, tf='NO', stringsAsFactors = FALSE), node_hop)
} else {
  node_hop$hop[node_hop$node == ligand] <- 1
}
if (!receptor %in% node_hop$node) {
  node_hop <- rbind(node_hop, data.frame(node=receptor, hop=2, tf='NO', stringsAsFactors = FALSE))
}

# assign y by hop
plot_node <- data.frame()
for (h1 in sort(unique(node_hop$hop))) {
  d <- node_hop[node_hop$hop == h1, ]
  d <- d[order(d$node), ]
  d$y <- seq_len(nrow(d))
  d$x <- h1
  plot_node <- rbind(plot_node, d)
}

# edge coordinates
src_xy <- plot_node[, c('node','x','y')]; colnames(src_xy) <- c('src','src_x','src_y')
dst_xy <- plot_node[, c('node','x','y')]; colnames(dst_xy) <- c('dst','dst_x','dst_y')
plot_edge <- edge_df %>% left_join(src_xy, by='src') %>% left_join(dst_xy, by='dst')

plot_node$Celltype <- receiver
plot_node$Celltype[plot_node$node == ligand] <- sender
plot_node$Celltype[plot_node$node == receptor] <- receiver
plot_node$Celltype <- factor(plot_node$Celltype, levels=c(sender, receiver))
plot_node$tf <- ifelse(plot_node$tf == 'YES', 'YES', 'NO')
plot_node$tf <- factor(plot_node$tf, levels = c('YES', 'NO'))

# style close to SpaVis plot_lr_path
filtered_data <- plot_node
p <- ggplot() +
  geom_segment(data = plot_edge, aes(x = src_x, y = src_y, xend = dst_x, yend = dst_y), linewidth = linewidth, color = '#cccccc') +
  geom_point(data = plot_node, aes(x, y, color = Celltype), size = 10) +
  scale_color_manual(values = c(sender_color, receiver_color)) +
  ggrepel::geom_label_repel(
    data = filtered_data,
    aes(x, y, label = node, fill = tf),
    size = label_size/3,
    max.overlaps = Inf,
    force = 2,
    box.padding = 0.5,
    point.padding = 0.3,
    min.segment.length = 0,
    seed = 1
  ) +
  scale_fill_manual(values = c('YES' = '#E74C3C', 'NO' = '#2ECC71')) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(clip = 'off') +
  theme(axis.text = element_blank(), panel.grid = element_blank(), axis.ticks = element_blank(), panel.background = element_rect(fill = 'white'),
        plot.margin = margin(10, 30, 10, 10),
        legend.text = element_text(size = 12), legend.title = element_text(size = 14))

sender_name <- gsub('/', '_', sender)
receiver_name <- gsub('/', '_', receiver)
out_pdf <- file.path(out_dir, paste0('alphavis_cascade_', sender_name, '_', receiver_name, '_', ligand, '_', receptor, '.pdf'))
out_svg <- file.path(out_dir, paste0('alphavis_cascade_', sender_name, '_', receiver_name, '_', ligand, '_', receptor, '.svg'))
ggsave(out_pdf, p, width = w, height = h)
tryCatch({ ggsave(out_svg, p, width = w, height = h) }, error=function(e){ message('Skip svg export: ', e$message) })

write.table(plot_edge, file=file.path(out_dir,'alphavis_cascade_edges.tsv'), sep='\t', quote=FALSE, row.names=FALSE)
write.table(plot_node, file=file.path(out_dir,'alphavis_cascade_nodes.tsv'), sep='\t', quote=FALSE, row.names=FALSE)
cat(out_pdf, '\n')
