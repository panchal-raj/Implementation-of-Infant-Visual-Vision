from properties import *


def main():
    age = int(input("Enter the age of the infant (in months): "))

    # Transformation for visual acuity
    print(f"{time.strftime('%H::%M::%S', time.localtime())}: Starting dataloader construction...")

    dataloader = DefineDataLoader(age=age)
    dataloader_va, dataloader_cs, dataloader_combined, dataloader_original = dataloader.construct_dataloader()

    print(f"{time.strftime('%H::%M::%S', time.localtime())}: Dataloader construction completed...")


    print(f"\n{time.strftime('%H::%M::%S', time.localtime())}: Dataloader runtime for VA...")
    # Visualize the results for visual acuity
    time_original, time_va = dataloader.dataloader_runtime(dataloader_original=dataloader_original,
                                                           dataloader_prop=dataloader_va,
                                                           property="Visual Acuity")


    print(f"{time.strftime('%H::%M::%S', time.localtime())}: Dataloader runtime for CS...")
    # Visualize the results for Contrast Sensitivity
    _, time_cs = dataloader.dataloader_runtime(dataloader_original=dataloader_original,
                                               dataloader_prop=dataloader_cs,
                                               property="Contrast Sensitivity")


    print(f"{time.strftime('%H::%M::%S', time.localtime())}: Dataloader runtime for both VA and CS...")
    # Visualize the results for combined transformations
    _, time_combined = dataloader.dataloader_runtime(dataloader_original=dataloader_original,
                                                     dataloader_prop=dataloader_combined,
                                                     property="both Transformations")

    # Visualize runtime of the dataloaders
    dataloader.visualize_runtime(time_original, time_va, time_cs, time_combined)

    # Visualize the transformed images in the dataloader
    dataloader.visualize_batch(dataloader_va, property="Visual Acuity")
    dataloader.visualize_batch(dataloader_cs, property="Contrast Sensitivity")
    dataloader.visualize_batch(dataloader_combined, property="Both transformation combined")


if __name__ == '__main__':
    main()
