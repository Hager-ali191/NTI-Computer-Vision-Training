


def download():
    rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
    project = rf.workspace(config.ROBOFLOW_WORKSPACE).project(config.ROBOFLOW_PROJECT)
    version = project.version(config.ROBOFLOW_VERSION)

    print(f"Downloading '{config.ROBOFLOW_PROJECT}' v{config.ROBOFLOW_VERSION} "
          f"in YOLOv8-seg format -> {config.DATA_DIR}")
    version.download(config.ROBOFLOW_FORMAT_YOLO, location=config.DATA_DIR)

    print(f"Downloading '{config.ROBOFLOW_PROJECT}' v{config.ROBOFLOW_VERSION} "
          f"in COCO-segmentation format -> {config.COCO_DATA_DIR}")
    version.download(config.ROBOFLOW_FORMAT_COCO, location=config.COCO_DATA_DIR)

    print("\nDone. Dataset structure:")
    print(f"  YOLO format : {config.DATA_DIR}/{{train,valid,test}}/{{images,labels}}, data.yaml")
    print(f"  COCO format : {config.COCO_DATA_DIR}/{{train,valid,test}}/_annotations.coco.json")


if __name__ == "__main__":
    download()
