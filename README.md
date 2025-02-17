## Official Repository of the paper titled "Active Geospatial Search for Efficient Tenant Eviction Outreach"

## Train 
To train our proposed method, please execute
```
Python train.py
```

## Model Architecture 
We use a Multi-head Transformer architecture as the prediction module. To use this model please set:
```
python train.py -- model MulT
```
For the search module, we develop a custom Neural Network. To use this model, please set:
```
python train.py -- model_search Model_search
```
We used 5 layers and 5 heads in the Transformer architecture. If you want to use a different number of layers and heads, please set the value accordingly in the argument:
```
python train.py -- model MulT -- model_search Model_search --nlevels 5 --num_heads 5
```

To customize the cross-modal fusion, set --aonly --vonly --lonly parameters accordingly.
## Eval

To test our proposed method, please execute
```
Python eval.py
```

Our code is adopted from:
https://github.com/yaohungt/Multimodal-Transformer

Please cite our paper if you find our work useful for your research:
```
@article{sarkar2024active,
  title={Active Geospatial Search for Efficient Tenant Eviction Outreach},
  author={Sarkar, Anindya and DiChristofano, Alex and Das, Sanmay and Fowler, Patrick J and Jacobs, Nathan and Vorobeychik, Yevgeniy},
  journal={arXiv preprint arXiv:2412.17854},
  year={2024}
}
```

