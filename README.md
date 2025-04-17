# MambaInterp
This repo is about Mamba Interpretability on the Binding Task

## Dataset Generation for Different Relation Format
To generate the dataset for different relation formats ("A loves B. C hates D. E adores F. . . . A loves" , you can use the following command:

```
python dataset_generation.py --n <number of discourse> --num_samples <number of sample for each discourse> --output_dir "dataset"
```

## Convert the Dataset Into the Same Relation
To convert the previously generated dataset into the same relation formats ("A loves B. C loves D. E loves F. . . . A loves"), you can run the following command:

```
python convert_dataset.py --input_dir <Input CSV file path> --output_dir <Output CSV file path>
```

## Generate the u-shaped figure
Then, you can evaluate the performance of the model in the generated dataset.
```
python ushape.py --model <hf-model> --datasets <"[list of dataset to be evaluated]"> --output_dir <output directory> --gpu 1
```

## Customized Mamba
In CustomizeMamba, I implement a function that replaces the original slow_forward method in the FalconMambaMixer class. This approach allows us to intercept and analyze intermediate values during the forward pass without directly modifying the Hugging Face transformers package. Important: Before using this code, make sure to uninstall the CUDA-optimized packages causal-conv1d and mamba-ssm. Their presence will cause the model to use the optimized implementation path instead of the slow_forward method we're instrumenting. I chose this non-invasive approach rather than directly modifying the Hugging Face source code to preserve the integrity of the package and avoid potential compatibility issues.
