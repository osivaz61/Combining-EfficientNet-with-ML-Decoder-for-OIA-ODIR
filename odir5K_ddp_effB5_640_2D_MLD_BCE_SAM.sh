#!/bin/bash
#SBATCH -p akya-cuda                                   # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A osivaz                                      # Kullanici adi
#SBATCH -J odir5K_ddp_effB5_640_2D_MLD_BCE_SAM         # Gonderilen isin ismi
#SBATCH -o odir5K_ddp_effB5_640_2D_MLD_BCE_SAM.out     # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:4                                   # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                                           # Gorev kac node'da calisacak?
#SBATCH -n 1                                           # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 40                             # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=15:00:00                                # Sure siniri koyun.

eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate python38
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 odir5K_ddp_effB5_640_2D_MLD_BCE_SAM.py
