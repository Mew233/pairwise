library(synergyfinder)
library(data.table)
library(dplyr)
library(tidyverse)

data_dir = getwd()
# Load the ALMANAC data.
tripod_dat = read.csv(paste0(data_dir, "/NCATSTable_btk_full.csv"))
# Convert it to a data.table. 
setDT(tripod_dat)

# create a block_id for groupby
tripod_dat <- tripod_dat %>% group_by(drug1) %>% mutate(block_id = cur_group_id())
screening_data <- tripod_dat %>% 
  dplyr::select(block_id,drug1,drug2,conc1, conc2, response) %>% 
  rename( "drug_row" = "drug1", 
          "drug_col" = "drug2",
          "conc_r" = "conc1",
          "conc_c" = "conc2", 
          "response"="response")

screening_data$conc_r <- screening_data$conc_r * 10^9
screening_data$conc_c <- screening_data$conc_c * 10^9
screening_data$conc_r_unit <- "nM"
screening_data$conc_c_unit <- "nM"
rm(tripod_dat)

#循环
empty_df = data.frame()

calsyn <- function(st,ed){
  test1 <- screening_data %>% filter(block_id %in% c(seq(st,ed))) #321240
  res <- synergyfinder::ReshapeData(
    data = test1,
    data_type = "viability",
    impute = TRUE,
    impute_method = NULL
  )
  res <- synergyfinder::CalculateSynergy(
    data = res,
    method = c("ZIP", "HSA", "Bliss", "Loewe"),
    Emin = NA,
    Emax = NA,
    correct_baseline = "non")
  
  empty_df <- rbind(empty_df,res$drug_pairs)
  empty_df
}

#>>>> Excute to calculate synergy score for each block of dose-response matrics
d <- 1:227#1:
lgth <- 1 # split length
chunks <- split(d, ceiling(seq_along(d)/lgth))

for (ck in chunks){
  st = head(ck,1)
  end = tail(ck,1)

  tryCatch(
    expr = {
      print(st)
      empty_df <- calsyn(st,end)
      message("Successfully executed the log(x) call.")
      return(empty_df)
    },
    error = function(e){
      message('Caught an error!')
      print(e)
    })
}
write.table(empty_df, file = "TripodDrugSynScores.csv",sep=",")