# 
# processing <- function(input, output){
#     
#     df     <- readr::read_delim(file = input, delim = " ", skip = 2, show_col_types = FALSE)
#     colnames(df) <- c("mass", "int")
#     df     <- dplyr::mutate(df, int = log(int+1))
#     model  <- stats::loess(int~mass, data=df, span = 0.0008)
#     df     <- dplyr::mutate(df, int = stats::predict(model, mass)) 
#     model  <- stats::loess(int~mass, data=df, span = 0.3)
#     df     <- dplyr::mutate(df, int =  int / stats::predict(model, df$mass))
#     df$int <- -1 + 2*(df$int-min(df$int))/(max(df$int)-min(df$int))
#     
#     utils::write.csv(file = output, x = df)
#     return(output)
# }



processing <- function(input, output){
    if (FALSE){ # Debug
        input <- "../maldi_preprocessing/DRIAMS-B/raw/2018/0a0bcba0-2d29-4e1b-a7b7-722b424b18bb.txt"
        output <- "../maldi_preprocessing/DRIAMS-B/preprocessed_raw/2018/0a0bcba0-2d29-4e1b-a7b7-722b424b18bb.txt"
    }
    df     <- readr::read_delim(file = input, delim = " ", skip = 3, col_names = c("mass", "int"))
    df     <- dplyr::mutate(df, int = log(int+1))
    model  <- stats::loess(int~mass, data=df, span = 0.0008)
    df     <- dplyr::mutate(df, int = stats::predict(model, mass)) 
    model  <- stats::loess(int~mass, data=df, span = 0.3)
    df     <- dplyr::mutate(df, int =  int / stats::predict(model, df$mass))
    df$int <- -1 + 2*(df$int-min(df$int))/(max(df$int)-min(df$int))
    
    # Map to a new fixed scale
    model  <- stats::loess(int~mass, data=df, span = 0.0003)
    mass   <- seq(2000, 20000, 0.5) 
    df     <- tibble::tibble(int = stats::predict(model, mass), mass = mass)
    
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


