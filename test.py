from Dataloaders.fundus_loader import download_class_fundus
from Dataloaders.cifar_loader import download_class_cifar
from FewShot_models.training_parallel import *
parser = get_arguments()
parser.add_argument('--dataset', help='cifar/mnist/fashionmnist/mvtec/paris', default='cifar')
parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
parser.add_argument('--pos_class', help='normal class',default='dog')
parser.add_argument('--random_images_download', help='random selection of images', default=False)
parser.add_argument('--num_images', type=int, help='number of images to train on', default=1)
parser.add_argument('--if_download', type=bool, help='do you want to download class', default=True)
parser.add_argument('--mode', help='task to be done', default='train')
parser.add_argument('--size_image', type=int, help='size orig image', default=128)
parser.add_argument('--num_epochs', type=int, help='num epochs', default=1)
parser.add_argument('--policy', default='')
parser.add_argument('--niter_gray', help='number of iterations in each scale', type=int, default=500)
parser.add_argument('--niter_rgb', help='number of iterations in each scale', type=int, default=1000)
parser.add_argument('--index_download', help='index in dataset for starting download', type=int, default=1)
parser.add_argument('--use_internal_load', help='using another dataset', default=False)
parser.add_argument('--experiment', help='task to be done', default='stop_signs')
parser.add_argument('--test_size', help='test size', type=int, default=10000)
parser.add_argument('--num_transforms', help='54 for rgb, 42 for grayscale', type=int, default=54)
parser.add_argument('--device_ids', help='gpus ids in format: 0/ 0 1/ 0 1 2..', nargs='+', type=int, default=0)
parser.add_argument('--fraction_defect', help='fraction of patches to consider in each scale', nargs='+', type=float, default=0.1)


opt = parser.parse_args()
download_class_cifar(opt)