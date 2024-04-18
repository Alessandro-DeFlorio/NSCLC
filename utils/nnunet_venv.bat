@echo off
:: Create the conda environment
call conda create --name nnunet python=3.8 -y

:: Activate the environment - Note: Direct activation doesn't work in scripts as expected, so we use other commands in the activated environment's context
call conda activate nnunet

:: Install cudatoolkit
call conda install cudatoolkit -c anaconda -y

:: Check NVIDIA GPU
call nvidia-smi

:: Install PyTorch, torchvision, torchaudio, and specific CUDA version for PyTorch
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

:: Install IPython kernel
call conda install ipykernel -y

:: Deactivate the environment
call conda deactivate

:: for C disk
set nnUNet_raw=C:/Users/aless/Desktop/git/NSCLC/data/nnUNet_raw
set nnUNet_preprocessed=C:/Users/aless/Desktop/git/NSCLC/data/nnUNet_preprocessed
set nnUNet_results=C:/Users/aless/Desktop/git/NSCLC/data/nnUNet_results

call nnUNetv2_plan_and_preprocess -d Dataset001_Apm --verify_dataset_integrity


call nnUNetv2_train 1 3d_lowres 0 -device cuda
call nnUNetv2_train 1 3d_lowres 1 -device cuda
call nnUNetv2_train 1 3d_lowres 2 -device cuda
call nnUNetv2_train 1 3d_lowres 3 -device cuda
call nnUNetv2_train 1 3d_lowres 4 -device cuda

call nnUNetv2_find_best_configuration 1 -c 3d_lowres 
call nnUNetv2_predict -d Dataset001_Apm -i C:/Users/aless/Desktop/git/NSCLC/data/nnUNet_raw/Dataset001_Apm/imagesTr -o C:/Users/aless/Desktop/git/NSCLC/data/magic -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_lowres -p nnUNetPlans

call nnUNetv2_apply_postprocessing -i C:/Users/aless/Desktop/git/NSCLC/data/magic -o C:/Users/aless/Desktop/git/NSCLC/data/final -pp_pkl_file C:/Users/aless/Desktop/git/NSCLC/data/nnUNet_results/Dataset001_Apm/nnUNetTrainer__nnUNetPlans__3d_lowres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json C:/Users/aless/Desktop/git/NSCLC/data/nnUNet_results/Dataset001_Apm/nnUNetTrainer__nnUNetPlans__3d_lowres/crossval_results_folds_0_1_2_3_4/plans.json

:: for D disk
set nnUNet_raw=D:/NSCLC/data/nnUNet_raw
set nnUNet_preprocessed=D:/NSCLC/data/nnUNet_preprocessed
set nnUNet_results=D:/NSCLC/data/nnUNet_results

call nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

call nnUNetv2_train 1 3d_lowres 0 -device cuda
call nnUNetv2_train 1 3d_lowres 1 -device cuda
call nnUNetv2_train 1 3d_lowres 2 -device cuda
call nnUNetv2_train 1 3d_lowres 3 -device cuda
call nnUNetv2_train 1 3d_lowres 4 -device cuda

call nnUNetv2_find_best_configuration 1 -c 3d_lowres
call nnUNetv2_predict -d Dataset001_Apm -i  D:/NSCLC/data/nnUNet_raw/Dataset001_Apm/imagesTs -o D:/NSCLC/data/magic -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_lowres -p nnUNetPlans

call nnUNetv2_apply_postprocessing -i D:/NSCLC/data/magic -o D:/NSCLC/data/final -pp_pkl_file D:/NSCLC/data/nnUNet_results/Dataset001_Apm/nnUNetTrainer__nnUNetPlans__3d_lowres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json D:/NSCLC/data/nnUNet_results/Dataset001_Apm/nnUNetTrainer__nnUNetPlans__3d_lowres/crossval_results_folds_0_1_2_3_4/plans.json


:: full res approach
set nnUNet_raw=D:/NSCLC/data/nnUNet_raw
set nnUNet_preprocessed=D:/NSCLC/data/nnUNet_preprocessed
set nnUNet_results=D:/NSCLC/data/nnUNet_results

call nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

call nnUNetv2_train 1 3d_fullres 0 -device cuda
call nnUNetv2_train 1 3d_fullres 1 -device cuda
call nnUNetv2_train 1 3d_fullres 2 -device cuda
call nnUNetv2_train 1 3d_fullres 3 -device cuda
call nnUNetv2_train 1 3d_fullres 4 -device cuda

call nnUNetv2_find_best_configuration 1 -c 3d_fullres
call nnUNetv2_predict -d Dataset001_Apm -i  D:/NSCLC/data/nnUNet_raw/Dataset001_Apm/imagesTs -o D:/NSCLC/data/magic -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans

call nnUNetv2_apply_postprocessing -i D:/NSCLC/data/magic -o D:/NSCLC/data/final -pp_pkl_file D:/NSCLC/data/nnUNet_results/Dataset001_Apm/nnUNetTrainer__nnUNetPlans__3d_lowres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json D:/NSCLC/data/nnUNet_results/Dataset001_Apm/nnUNetTrainer__nnUNetPlans__3d_lowres/crossval_results_folds_0_1_2_3_4/plans.json

: for 2d
set nnUNet_raw=D:/NSCLC/data/nnUNet_raw
set nnUNet_preprocessed=D:/NSCLC/data/nnUNet_preprocessed
set nnUNet_results=D:/NSCLC/data/nnUNet_results

call nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

call nnUNetv2_train 3 3d_lowres 0 -device cuda
call nnUNetv2_train 3 3d_lowres 1 -device cuda
call nnUNetv2_train 3 3d_lowres 2 -device cuda
call nnUNetv2_train 3 3d_lowres 3 -device cuda
call nnUNetv2_train 3 3d_lowres 4 -device cuda

call nnUNetv2_find_best_configuration 3 -c 3d_lowres

call nnUNetv2_predict -d Dataset001_Apm -i D:\nsclc\data\nnUNet_raw\Dataset001_Apm\imagesTs -o D:\nsclc\data\nnUNet_raw\Dataset001_Apm\outputTs -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans
call nnUNetv2_apply_postprocessing -i D:\script_di_ale\output\lungMask -o D:\script_di_ale\output\lungMaskPP -pp_pkl_file D:/NSCLC/data/nnUNet_results\Dataset001_Apm\nnUNetTrainer__nnUNetPlans__2d\crossval_results_folds_0_1_2_3_4\postprocessing.pkl -np 8 -plans_json D:/NSCLC/data/nnUNet_results\Dataset001_Apm\nnUNetTrainer__nnUNetPlans__2d\crossval_results_folds_0_1_2_3_4\plans.json

call nnUNetv2_predict -d Dataset001_Apm -i D:\nsclc\data\nnUNet_raw\Dataset003_Lung\imagesTr -o D:\nsclc\data\nnUNet_raw\Dataset003_Lung\outputTs -f  0 -tr nnUNetTrainer -c 3d_lowres -p nnUNetPlans

