#################################################
#Figure 4: Heat Map of Manual and Modeled Poultry Locations.
#################################################
  rm(list=ls())

  #Assume base R is running
  my.wd <- getSrcDirectory(function(x){x})

  #Check for errors
  if(length(my.wd)==0){
    #Assume current directory is working directory
    my.wd <- "."
  }else if(grepl("error", tolower(class(my.wd)[1]))|my.wd==""){
    #Try to access working directory through R Studio API
    my.wd <- tryCatch(dirname(rstudioapi::getActiveDocumentContext()$path),
                      error = function(e) e)
  }else{
    my.wd <- "."
  }

  #Set working directory
  setwd(my.wd)

  #Source function to install packages
  source("functions/installPackageNotFound.R")

  #Packages
  installPackageNotFound("rgdal")
  installPackageNotFound("RColorBrewer")
  installPackageNotFound("sp")
  installPackageNotFound("scales")
  installPackageNotFound("showtext")

  #Add font to match python figures
  font_add(family = "DejaVu", regular = "../data/fonts/DejaVuSans.ttf",
           bold = "../data/fonts/DejaVuSans-Bold.ttf")

  #Enable showtext to set font
  showtext_auto()

#################################################
#Read in shapefiles
#################################################

  #Read in census block shapefile
  shp.file <- readOGR(
    "../data/shapefiles/cb_2017_37_bg_500k/cb_2017_37_bg_500k.shp",
                      "cb_2017_37_bg_500k")

  #Set projection system
  shp.file <- spTransform(shp.file,
    CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))

#################################################
#Read in facility data
#################################################

  #Read in CAFO manually validated classes
  cafo.locations <- read.csv("../data/csv/facility_accuracy_data.csv",
  stringsAsFactors = F)

  #Prepare modeled data
  all.cafos <- cafo.locations[,c("facility_id", "latitude", "longitude",
  "true_positive", "new_cafo")]
  all.cafos$animal <- "POULTRY"
  all.cafos$source <- "model"
  all.cafos <- all.cafos[,c("facility_id", "latitude", "longitude", "animal",
  "true_positive", "source", "new_cafo")]

  #Set coords for cafo locations
  all.cafos <- SpatialPointsDataFrame(all.cafos[,c("longitude", "latitude")],
  all.cafos, proj4string = CRS(proj4string(shp.file)))

  #Match locations to census blocks
  cafo.blocks <- data.frame("GEOID" = over(all.cafos, shp.file)[,c("GEOID")],
  stringsAsFactors = F)

  #Merge in demos
  all.cafos@data <- cbind(all.cafos@data, cafo.blocks)

  #Add all positive
  all.cafos$all_positive <- 1

  #Aggregate to block level
  block.agg <- aggregate(cbind(all_positive, true_positive, new_cafo)~
  GEOID+source+animal, data = all.cafos, FUN = sum)

  #Widen
  block.agg <- reshape(block.agg, direction = "wide",
  idvar = c("GEOID", "source"), timevar = "animal")

  #Replace NA's with 0's
  cols <- paste(c("all_positive", "true_positive", "new_cafo"),
  "POULTRY", sep =".")
  for(this.col in cols){
    block.agg[is.na(block.agg[,this.col]), this.col] <- 0
  }

#################################################
#Map of cafo locations
#################################################

  pdf("../figures/4_Figure_fgmap_Manual_Modeled_Poultry_Locations.pdf",
  width = 8.5, height = 8)
    layout(matrix(c(1,1,1,3,1,1,2,2,2,4,2,2),4,3,byrow=T),
    width=c(0.4,0.5,0.5), height=c(0.4,0.16,0.4,0.16))
    par(mar=c(0,0.25,0,0.5), mai = c(0,0,0,0), tcl=-0.3,
        pin = c(3,3),
        pty = "m",
        xaxt = 'n',
        xpd = TRUE,
        yaxt = 'n',
        family = 'DejaVu')

    #Create map for manual, modeled, and new locations
    for(this.source in c("model", "new")){

      #Source data
      source.data <- "model"

      #Heatmap column
      heatmap.column <- ifelse(this.source%in%c("model"),
      "all_positive.POULTRY", "new_cafo.POULTRY")

      #Values
      if(this.source%in%c("model")){
        heatmap.values <- c(0,1,2,4,6,8,10,12,14,max(block.agg[,
          heatmap.column])+1)
      }else{
        heatmap.values <- c(0,1,2,3,max(block.agg[,heatmap.column])+1)
      }

      #Save object to global
      assign(paste0("color.values.",heatmap.column), heatmap.values)

      #Modeled map
      shp.file.model <- merge(shp.file,
        block.agg[which(block.agg$source==source.data),], by = "GEOID",
        all.x = T)
      shp.file.model@data[is.na(shp.file.model@data[,heatmap.column]),
      heatmap.column] <- 0

      #Color bins
      shp.file.model@data$cafo_group <- cut(shp.file.model@data[,heatmap.column],
                                        heatmap.values,
                                        right = F)
      #Color wheel
      color.ramp <- brewer.pal(length(levels(shp.file.model@data$cafo_group)),
      "YlOrRd")
      assign(paste0("color.ramp.",heatmap.column), color.ramp)

      #Heatmap
      plot(shp.file.model,
        col = color.ramp[as.numeric(shp.file.model@data$cafo_group)],
       border = alpha("#4d4d4d", 0.2), lwd = 0.3,
       xlim=c(-81.3, -78.4), ylim=c(34-0.25, 37.2- 0.25))

      #Determine title
      this.title <- ifelse(this.source=="model", "Modelled locations",
      "New locations")

      #Set title
      title(this.title, line = -2,cex.main=2)

    }

    #Graphics params for legend 1
    par(mar = c(5,5,3,1),
    mgp=c(1.5,0.5,0), tcl=-0.3,
    xaxt = 's',
    xpd = TRUE,
    yaxt = 's',
    family = 'DejaVu')

    #Create first legend
    image(color.values.all_positive.POULTRY, 1,
      as.matrix(seq_along(color.values.all_positive.POULTRY)),
    col=color.ramp.all_positive.POULTRY, xlab="Number of CAFOs",ylab="", axes=F)
    axis(1)

    #Graphics params for legend 2
    par(mar = c(5,5,3,1),
    mgp=c(1.5,0.5,0), tcl=-0.3,
    xaxt = 's',
    xpd = TRUE,
    yaxt = 's',
    family = 'DejaVu')

    #Create second legend
    image(color.values.new_cafo.POULTRY, 1,
      as.matrix(seq_along(color.values.new_cafo.POULTRY)),
    col=color.ramp.new_cafo.POULTRY, xlab="Number of New CAFOs",ylab="", axes=F)
    axis(1)

  dev.off()
