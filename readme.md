# Importan

If you haven't cuda devices on machine, install tf version without +cu postfix (look at the requiement file)

# Server side background removal

Upload video and image and get hologram as a result

## Run service

1. [Install docker](https://docs.docker.com/v17.09/engine/installation/)


2. Check if there's a DS model
    ```text
    ...
    ├── content
        └── pytorch_resnet50.pth
   ...
    ``` 
   Download from [Google Drive](https://drive.google.com/drive/mobile/folders/1cbetlrKREitIgjnIikG1HdM4x72FtgBh?usp=sharing)
   
   TODO: ask @Disa add it to Git LSF


3. Run  service
    ```bash
    cd /path/to/service-directory
    docker-compose up -d --build 
    ``` 

4. Upload files [here](http://localhost:9090/upload) to remove bg

