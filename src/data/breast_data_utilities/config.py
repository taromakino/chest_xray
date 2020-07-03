number_of_classes = 3
number_of_views = 4
number_of_density_level = 4
number_of_cancer_classes = 2
number_of_cancer_classes_with_unknown = 3
names_of_classes = ['BI-RADS 0', 'BI-RADS 1', 'BI-RADS 2']
names_of_density = ['almost entirely fatty (0)' , 'scattered areas of fibroglandular density (1)', 'heterogeneously dense (2)','extremely dense (3)']

input_size = (2600, 2000)
data_buffer_size = 30
metadata_buffer_size = 100
number_of_loaders = 5

input_format = 'binary'

features_used = 'only image'
non_image_features = ['age']
tracing_performance = False

input_size_dict = {'MLO': (2974, 1748), 'CC': (2677, 1942)}

