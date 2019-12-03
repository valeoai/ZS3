import valeodata

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            #return str(valeodata.download('zsd_pascalVOC/VOC2012'))
            return '/datasets_local/zsd_pascalVOC/VOC2012'
        elif dataset == 'sbd':
            #return str(valeodata.download('zsd_pascalVOC/VOC2012/benchmark_RELEASE'))
            return '/datasets_local/zsd_pascalVOC/VOC2012/benchmark_RELEASE'
        elif dataset == 'context':
            valeodata.download('zsd_segmentation_pascalVOC/datasets/context')
            return '/datasets_local/zsd_segmentation_pascalVOC/datasets/context'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError