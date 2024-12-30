library(tidyverse)
processing <- function(input, output){
  if (FALSE){ # Debug
    input <- "../data/DRIAMS-B/raw/2018/0a0bcba0-2d29-4e1b-a7b7-722b424b18bb.txt"
    output <- "../data/DRIAMS-B/raw_end_to_end/2018/0a0bcba0-2d29-4e1b-a7b7-722b424b18bb.txt"
  }
  df     <- readr::read_delim(file = input, delim = " ", skip = 3, col_names = c("mass", "int"))
  df$int <- -1 + 2*(df$int-min(df$int))/(max(df$int)-min(df$int))
  
  # Map to a new fixed scale
  df <- df |> filter(mass > 2000) |> slice(1:20000)
  readr::write_csv(df, output)
  return()
}

args <- commandArgs(trailingOnly = TRUE)
arg1 <- args[1]
arg2 <- args[2]

# Create the directory if it doesn't exist
if (!file.exists(arg2)) {
  dir.create(dirname(arg2), recursive = TRUE)
}

processing(arg1, arg2)


