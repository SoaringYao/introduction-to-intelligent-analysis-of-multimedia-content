## 主体结构

```
├───.idea                   #项目配置文件，pycharm自行生成
│   └───inspectionProfiles
├───.run                    #配置好的 run configuration
├───figures                 #输入输出图片
└───Writerside              #程序文档
    ├───cfg
    ├───images              #文档插图
    └───topics              #文档内容
```

## 目录结构

```
│   fh.log              #利用logging生成的程序运行日志
│   sift.py             #代码
│   readme.md           #简介
│   
├───.idea               #IDE生成的配置文件
│   │   .gitignore
│   │   misc.xml
│   │   modules.xml
│   │   other.xml
│   │   sift_orb.iml
│   │   vcs.xml
│   │   workspace.xml
│   │   
│   └───inspectionProfiles
│           profiles_settings.xml
│           Project_Default.xml
│           
├───.run                #四种运行配置
│       default-type.run.xml
│       dry-run.run.xml
│       extreme-type.run.xml
│       tolerant-type.run.xml
│       
├───figures             #input0/1为输入图片，sift为输出图片
│       input0.png
│       input1.png
│       sift_d.png
│       sift_e.png
│       sift_t.png
│       
└───Writerside          #程序文档主文件夹(利用Writerside完成)
    │   c.list
    │   sift.tree
    │   v.list
    │   writerside.cfg
    │   
    ├───cfg
    │       buildprofiles.xml
    │       
    ├───images
    │       match_d.png
    │       match_e.png
    │       match_t.png
    │       
    └───topics          #详细的markdown文档内容
            starter-topic.md
```

## 说明

1. 程序源代码在[此处](./sift.py)
2. 详细的程序介绍和结果分析在[此处](./Writerside/topics/starter-topic.md)
3. 运行代码的环境为 python 3.11, 依赖的库见[此处](./requirements.txt)
