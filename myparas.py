class getparas(object):
    @staticmethod
    def mypath():
        # return "/Users/karax/Desktop/1 Lab/2 Drone and parking/2Runs/pointcloud/CAINZ_下妻店/output/pieces_5m"
        # return "/Users/karax/Desktop/1 Lab/2 Drone and parking/2Runs/pointcloud/CAINZ_相馬店/Output_cross/crop_5m"
        return "/Users/karax/Desktop/1 Lab/2 Drone and parking/2Runs/pointcloud/tokoname_down/metashape_project/output"
    @staticmethod
    def myparas():
        return {
            # Parameters
            'Downsampling_voxel_size': 0.01,

            # Key parameters
            'Plane_seg_threshold': 0.01,
            'Refine_size': 7000,
            'Refine_times': -1,
            'clustering_eps': 0.1,
            'Ransac_n': 5,
            'OriShow': True
        }
