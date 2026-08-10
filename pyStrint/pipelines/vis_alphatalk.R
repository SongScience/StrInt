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
top_n <- as.integer(args[13])

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(dplyr)
})

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

x <- read.delim(in_tsv, sep='\t', header=TRUE, stringsAsFactors = FALSE)
if (!all(c('ligand', 'receptor', 'downstream_gene') %in% colnames(x))) {
  stop('Input must contain ligand/receptor/downstream_gene columns.')
}

x <- x %>% distinct(ligand, receptor, downstream_gene)
if (nrow(x) == 0) stop('No rows in alphatalk_lr_path.tsv')
if (nrow(x) > top_n) x <- x[1:top_n, ]

lr_node <- paste0(ligand, '|', receptor)

layer1 <- data.frame(node = ligand, x = 1, y = 1, type='SenderLigand', stringsAsFactors = FALSE)
layer2 <- data.frame(node = receptor, x = 2, y = 1, type='ReceiverReceptor', stringsAsFactors = FALSE)

down <- x %>%
  distinct(downstream_gene) %>%
  mutate(node = downstream_gene,
         x = 3,
         y = row_number(),
         type = 'Downstream') %>%
  select(node, x, y, type)

if (nrow(down) == 1) {
  down$y <- 1
}

nodes <- bind_rows(layer1, layer2, down)

edges <- bind_rows(
  data.frame(src = ligand, dst = receptor, stringsAsFactors = FALSE),
  x %>% transmute(src = receptor, dst = downstream_gene)
) %>% distinct()

src_xy <- nodes %>% select(node, src_x = x, src_y = y)
dst_xy <- nodes %>% select(node, dst_x = x, dst_y = y)

edges_plot <- edges %>%
  left_join(src_xy, by = c('src' = 'node')) %>%
  left_join(dst_xy, by = c('dst' = 'node')) %>%
  transmute(x = src_x, y = src_y, xend = dst_x, yend = dst_y)

color_map <- c(sender_color, receiver_color, '#8f8f8f')
names(color_map) <- c('SenderLigand', 'ReceiverReceptor', 'Downstream')

p <- ggplot() +
  geom_segment(data = edges_plot, aes(x = x, y = y, xend = xend, yend = yend),
               linewidth = linewidth, color = '#cccccc') +
  geom_point(data = nodes, aes(x = x, y = y, color = type), size = 10) +
  scale_color_manual(values = color_map) +
  ggrepel::geom_label_repel(
    data = nodes,
    aes(x = x, y = y, label = node, fill = type),
    size = label_size / 3,
    max.overlaps = Inf,
    box.padding = 0.25,
    point.padding = 0.15
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = paste0(sender, ': ', ligand, ' -> ', receiver, ': ', receptor, ' -> downstream')
  ) +
  theme_bw() +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    panel.background = element_rect(fill = 'white'),
    legend.title = element_blank()
  )

sender_name <- gsub('/', '_', sender)
receiver_name <- gsub('/', '_', receiver)
out_pdf <- file.path(out_dir, paste0('alphavis_lr_path_', sender_name, '_', receiver_name, '_', ligand, '_', receptor, '.pdf'))
out_svg <- file.path(out_dir, paste0('alphavis_lr_path_', sender_name, '_', receiver_name, '_', ligand, '_', receptor, '.svg'))

ggsave(out_pdf, p, width = w, height = h)
tryCatch({
  ggsave(out_svg, p, width = w, height = h)
}, error = function(e) {
  message('Skip svg export: ', e$message)
})

cat(out_pdf, '\n')
cat(out_svg, '\n')
