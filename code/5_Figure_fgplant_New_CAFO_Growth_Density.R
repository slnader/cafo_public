#################################################
#Figure 5: Longitudinal Detection of CAFO Growth
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
  installPackageNotFound("ggmap") 
  installPackageNotFound("sp")
  installPackageNotFound("rgdal") 
  installPackageNotFound("getPass")
  installPackageNotFound("showtext")
  
  #Add font to match python figures
  font_add(family = "DejaVu", regular = "../data/fonts/DejaVuSans.ttf", 
           bold = "../data/fonts/DejaVuSans-Bold.ttf")
  
  #Enable showtext to set font
  showtext_auto()
  
  #Ask for API key to use Google Maps
  api.key <- getPass(msg="Please provide your Google Maps API Key:")
  register_google(key = api.key)
  
#################################################
#Read in model results
#################################################
  #Model results
  facility.data <- read.csv("../data/csv/longitudinal_image_data.csv", stringsAsFactors = F)
  
  #Point type
  facility.data$point.type <- "Existing CAFO"
  facility.data$point.type[which(facility.data$true_naip_year>=2011 & facility.data$image_id!="Poultry Plant")] <- "New CAFO"
  facility.data$point.type[which(facility.data$image_id=="Poultry Plant")] <- "Plant Location"
  facility.data$point.type <- as.factor(facility.data$point.type)
  
#################################################
#Map of facility locations after plant opens
#################################################
  
  #Location of the processing plant
  plant.coords <- c(-77.669392,35.259436)

  #get map from open street
  city_map <- get_map(location = c(lon = plant.coords[1], lat =  plant.coords[2]), zoom = 9,
                                        maptype = "hybrid", api_key = api.key)
  
  #Separate manual and modeled data
  manual.data <- facility.data
  manual.data$dataset <- "Manual"
  
  #Modeled data
  modeled.data <- facility.data[!is.na(facility.data$modeled_naip_year),]
  modeled.data$dataset <- "Modelled"
  
  #Combine datasets
  master.map.data <- rbind(manual.data, modeled.data)
  
  #Create Map
  this.map <- ggmap(city_map) + 
    stat_density2d( data = master.map.data[which(master.map.data$point.type=="New CAFO"),], 
                    geom="polygon", 
                  aes( x = longitude, y = latitude, fill = ..level.. , alpha = ..level..) ) +
    geom_point(x=plant.coords[1], y = plant.coords[2], col="white")+
    geom_text(x=plant.coords[1], y = plant.coords[2]-0.05, label = "Feed mill", size = 1.5, col="white",
              family = 'DejaVu')+
    theme(axis.text.x= element_blank() ,
          axis.text.y= element_blank() , axis.ticks.x= element_blank() ,
          axis.ticks.y= element_blank() , axis.title = element_blank(),
          legend.key.size = unit(0.2, "cm"),
          legend.title = element_text(face="plain", family = 'DejaVu'),
          legend.position = "bottom",
          legend.key=element_blank(),
          legend.background= element_blank(),
          legend.text = element_text(size=5),
          legend.margin=margin(t=-0.3, r=0, b=-0.3, l=0, unit="cm"),
          title = element_text(size=7, face='bold', family = 'DejaVu'),
          plot.title = element_text(margin=margin(0,0,-1.5,0)),
          strip.background = element_blank(),
          plot.margin = unit(c(-0.7,0.1,-0.5,0), "cm"),
          strip.text.x = element_text(family = 'DejaVu'))+
    scale_fill_continuous( low = "yellow", high = "red", name = "New CAFO Density")+
    scale_alpha(range = c(0, 0.8), guide = FALSE)+
        guides(fill = guide_colourbar(ticks = FALSE, label = FALSE))+
    facet_wrap(~dataset)+
    ggtitle("Poultry CAFO Growth")

  #Save figure
  ggsave("../figures/5_Figure_fgplant_New_CAFO_Growth_Density.pdf",
         plot = this.map, width = 3.5, height = 2.3, units = "in")
  