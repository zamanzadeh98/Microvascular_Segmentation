#  Segment instances of microvascular structures from healthy human kidney tissue slides.

**Why?**
The proper operation of your body's organs and tissues relies on the interaction, spatial arrangement, and specialization of your cells, and there are a staggering 37 trillion of them. With such a vast number of cells, comprehending their functions and relationships represents a monumental task.
Ongoing initiatives to chart these cells involve the Vasculature Common Coordinate Framework (VCCF), which utilizes the human body's blood vasculature as the primary navigation system. The VCCF spans across all levels, ranging from the entire body down to the individual cell level, providing a distinctive means to pinpoint cellular locations using capillary structures as an addressing system. However, gaps in our understanding of microvasculature create gaps in the VCCF itself. If we could automatically segment microvasculature patterns, researchers would be able to leverage real-world tissue data to begin bridging these gaps and constructing a comprehensive map of the vasculature

## Example
![Alt](pictures/first/0.png)
![Alt Text](pictures/first/1.png)
![Alt Text](pictures/first/2.png)



## Models:
UNET (base line)
Transfer learning (Hopefully in the near future!)
Transformer (Hopefully in the near future!)
...


## How to use?
**Download**
```git clone https://github.com/zamanzadeh98/Microvascular_Segmentation.git```

**Installing packages**
!pip install requirements.txt```

Then, you can simply replace the model and image paths in the following scripts
```python your_script.py /path/to/your/model.pth /path/to/your/data.jpg```



[Dataset](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data)

