# Experiment 1 Log

## Translation of Flickr30K with NLLB

အောက်ပါဖိုင်ကို မြန်မာဘာသာပြောင်းမယ်။  

```
(base) ye@lst-hpc3090:~/intern3/img2txt/data$ head captions.txt
1000268201_693b08cb0e.jpg#0     A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg#1     A girl going into a wooden building .
1000268201_693b08cb0e.jpg#2     A little girl climbing into a wooden playhouse .
1000268201_693b08cb0e.jpg#3     A little girl climbing the stairs to her playhouse .
1000268201_693b08cb0e.jpg#4     A little girl in a pink dress going into a wooden cabin .
1001773457_577c3a7d70.jpg#0     A black dog and a spotted dog are fighting
1001773457_577c3a7d70.jpg#1     A black dog and a tri-colored dog playing with each other on the road .
1001773457_577c3a7d70.jpg#2     A black dog and a white dog with brown spots are staring at each other in the street .
1001773457_577c3a7d70.jpg#3     Two dogs of different breeds looking at each other on the road .
1001773457_577c3a7d70.jpg#4     Two dogs on pavement moving toward each other .
```

စာကြောင်ရေ စုစုပေါင်း လေးသောင်းရှိတယ်။  

```
(base) ye@lst-hpc3090:~/intern3/img2txt/data$ wc ./captions.txt
  40460  517166 3395237 ./captions.txt
```

nllb data path အောက်ကို ကော်ပီကူးယူခဲ့...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/flickr30k$ ls
captions.txt
```

## Cutting  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/flickr30k$ cut -f 2 ./captions.txt > f2.en
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/flickr30k$ head f2.en
A child in a pink dress is climbing up a set of stairs in an entry way .
A girl going into a wooden building .
A little girl climbing into a wooden playhouse .
A little girl climbing the stairs to her playhouse .
A little girl in a pink dress going into a wooden cabin .
A black dog and a spotted dog are fighting
A black dog and a tri-colored dog playing with each other on the road .
A black dog and a white dog with brown spots are staring at each other in the street .
Two dogs of different breeds looking at each other on the road .
Two dogs on pavement moving toward each other .
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/flickr30k$
```

Check with tail command...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/flickr30k$ tail ./f2.en
A person stands near golden walls .
a woman behind a scrolled wall is writing
A woman standing near a decorated wall writes .
The walls are covered in gold and patterns .
Woman writing on a pad in room with gold , decorated walls .
A man in a pink shirt climbs a rock face
A man is rock climbing high in the air .
A person in a red shirt climbing up a rock face covered in assist handles .
A rock climber in a red shirt .
A rock climber practices on a rock climbing wall .
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/flickr30k$
```

## Prepare Shell Script for Translation  

(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb$ cat flickr30k2my.sh  

```bash
#!/bin/bash

# Written by Ye Kyaw Thu, LU Lab., Myanmar
# Last Update: 25 June 2025

# Base directory for input files
INPUT_DIR="/home/ye/ye/exp/gpt-mt/nllb/data/flickr30k/"

# Directory for output files
OUTPUT_DIR="/home/ye/ye/exp/gpt-mt/nllb/flickr30k/"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over each .src file in the input directory
for FILE in "$INPUT_DIR"/*.en; do
    # Extract the base filename without the extension
    BASENAME=$(basename "$FILE" .en)

    # Define the output file name
    OUTPUT_FILE="$OUTPUT_DIR/$BASENAME.my"

    # Print the command being executed (for debugging)
    echo "Running nllb-translate.sh for $FILE"

    # Run the translation command
    time ./nllb-translate.sh --input "$FILE" --source eng_Latn --target mya_Mymr --output "$OUTPUT_FILE"
done

```

## Translation of Flickr30K   

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb$ ./flickr30k2my.sh
Running nllb-translate.sh for /home/ye/ye/exp/gpt-mt/nllb/data/flickr30k//f2.en
JSON Payload: {
  "text": "A child in a pink dress is climbing up a set of stairs in an entry way .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ပန်းရောင်ဝတ်စစုံဝတ်ထားတတဲ့ ကလေးတစ်ယောက်က ဝင်ပေါက်လမ်းမှာ လှေကားတစ်စီးတက်နေတယ်။"}
JSON Payload: {
  "text": "A girl going into a wooden building .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"သစ်သားအဆောက်အအအုံထဲ ဝင်နေတတဲ့ မိန်းကလေး။"}
JSON Payload: {
  "text": "A little girl climbing into a wooden playhouse .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"သစ်သားကစားရရုံထဲ တက်လာတတဲ့ ကောင်မလေးတစ်ယောက်"}
JSON Payload: {
  "text": "A little girl climbing the stairs to her playhouse .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"သူမရရဲ့ ကစားခန်းဆီ လှေကားထစ်တက်နေတတဲ့ ကလေးမလေးတစ်ယောက်"}
JSON Payload: {
  "text": "A little girl in a pink dress going into a wooden cabin .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ပန်းရောင်ဝတ်စစုံနနဲ့ သစ်သားအိမ်လေးထဲ ဝင်နေတတဲ့ မိန်းကလေးလေး။"}
JSON Payload: {
  "text": "A black dog and a spotted dog are fighting",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ခွေးမည်းနနဲ့ ခွေးအစက်တွေ တတိုက်နေကြတာ"}
JSON Payload: {
  "text": "A black dog and a tri-colored dog playing with each other on the road .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ခွေးမည်းနနဲ့ ခွေးသသုံးမျျိုး လမ်းပေါ်မှာ ကစားနေကြတယ်။"}
JSON Payload: {
  "text": "A black dog and a white dog with brown spots are staring at each other in the street .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"အနက်ရောင်ခွေးနနဲ့ အညညိုရောင်အစက်တွေပါတတဲ့ အဖြူရောင်ခွေဟာ လမ်းပေါ်မှာ အချင်းချင်း ငေးကြည့်နေကြတယ်။"}
JSON Payload: {
  "text": "Two dogs of different breeds looking at each other on the road .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"လမ်းပေါ်မှာ မတူညီတတဲ့ ခွေးနှစ်ကောင် အချင်းချင်းကြည့်နေတာပါ။"}
JSON Payload: {
  "text": "Two dogs on pavement moving toward each other .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"လမ်းဘေးမှာ ခွေးနှစ်ကောင် အချင်းချင်း ချဉ်းကပ်နေကြတယ်။"}
JSON Payload: {
  "text": "A little girl covered in paint sits in front of a painted rainbow with her hands in a bowl .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ပန်းချီနနဲ့ ဖဖုံးအုပ်ထားတတဲ့ မိန်းကလေးလေးက ပန်းချီဆွဲထားတတဲ့ သက်တန့်တစ်ပွင့်ရှေ့မှာ ထထိုင်နေပြီး လက်တွေကကို အိုးထဲမှာ ထည့်ထားတယ်။"}
JSON Payload: {
  "text": "A little girl is sitting in front of a large painted rainbow .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ကလေးမလေးက ပန်းချီဆွဲထားတတဲ့ သက်တန့်ကြီးရှေ့မှာ ထထိုင်နေတယ်။"}
JSON Payload: {
  "text": "A small girl in the grass plays with fingerpaints in front of a white canvas with a rainbow on it .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"မြက်ခင်းထဲမှာရှိတတဲ့ ကောင်မလေးက မမိုးကောင်းကင်နနဲ့အတူ အဖြူရောင် ပတ္တူစရှေ့မှာ လက်ချောင်းဆေးတွေနနဲ့ ကစားနေတယ်။"}
...
...
```

screen command ခံထားလို့ အထက်မှာ မြန်မာစလုံးမှာ အမှားအနေနဲ့ မြင်ရတာတွေ ရှိပေမဲ့...
Flickr30k ဒေတာရဲ့ အင်္ဂလိပ်စာ စာကြောင်းတွေက တိုတာရယ် လွယ်တာရယ်ကြောင့် မြန်မာလို translation လုပ်တာက အမှားနည်းမယ်လို့ ယူဆတယ်။  

screen command ဖြုတ်ပြီး ကြည့်ရင် အောက်ပါအတိုင်း ...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/flickr30k$ cat f2.my
A child in a pink dress is climbing up a set of stairs in an entry way .        ပန်းရောင်ဝတ်စုံဝတ်ထားတဲ့ ကလေးတစ်ယောက်က ဝင်ပေါက်လမ်းမှာ လှေကားတစ်စီးတက်နေတယ်။
A girl going into a wooden building .   သစ်သားအဆောက်အအုံထဲ ဝင်နေတဲ့ မိန်းကလေး။
A little girl climbing into a wooden playhouse .        သစ်သားကစားရုံထဲ တက်လာတဲ့ ကောင်မလေးတစ်ယောက်
A little girl climbing the stairs to her playhouse .    သူမရဲ့ ကစားခန်းဆီ လှေကားထစ်တက်နေတဲ့ ကလေးမလေးတစ်ယောက်
A little girl in a pink dress going into a wooden cabin .       ပန်းရောင်ဝတ်စုံနဲ့ သစ်သားအိမ်လေးထဲ ဝင်နေတဲ့ မိန်းကလေးလေး။
A black dog and a spotted dog are fighting      ခွေးမည်းနဲ့ ခွေးအစက်တွေ တိုက်နေကြတာ
A black dog and a tri-colored dog playing with each other on the road . ခွေးမည်းနဲ့ ခွေးသုံးမျိုး လမ်းပေါ်မှာ ကစားနေကြတယ်။
A black dog and a white dog with brown spots are staring at each other in the street .  အနက်ရောင်ခွေးနဲ့ အညိုရောင်အစက်တွေပါတဲ့ အဖြူရောင်ခွေဟာ လမ်းပေါ်မှာ အချင်းချင်း ငေးကြည့်နေကြတယ်။
Two dogs of different breeds looking at each other on the road .        လမ်းပေါ်မှာ မတူညီတဲ့ ခွေးနှစ်ကောင် အချင်းချင်းကြည့်နေတာပါ။
Two dogs on pavement moving toward each other . လမ်းဘေးမှာ ခွေးနှစ်ကောင် အချင်းချင်း ချဉ်းကပ်နေကြတယ်။
A little girl covered in paint sits in front of a painted rainbow with her hands in a bowl .    ပန်းချီနဲ့ ဖုံးအုပ်ထားတဲ့ မိန်းကလေးလေးက ပန်းချီဆွဲထားတဲ့ သက်တန့်တစ်ပွင့်ရှေ့မှာ ထိုင်နေပြီး လက်တွေကို အိုးထဲမှာ ထည့်ထားတယ်။
A little girl is sitting in front of a large painted rainbow .  ကလေးမလေးက ပန်းချီဆွဲထားတဲ့ သက်တန့်ကြီးရှေ့မှာ ထိုင်နေတယ်။
A small girl in the grass plays with fingerpaints in front of a white canvas with a rainbow on it .     မြက်ခင်းထဲမှာရှိတဲ့ ကောင်မလေးက မိုးကောင်းကင်နဲ့အတူ အဖြူရောင် ပတ္တူစရှေ့မှာ လက်ချောင်းဆေးတွေနဲ့ ကစားနေတယ်။
There is a girl with pigtails sitting in front of a rainbow painting .  မိုးကုလားအုပ် ပန်းချီကားရှေ့မှာ ထိုင်နေတဲ့ ဆံပင်အရှည်နဲ့ မိန်းကလေးတစ်ယောက်ရှိတယ်
Young girl with pigtails painting outside in the grass .        အပြင်မှာ မြက်ခင်းထဲမှာ ပန်းချီဆွဲနေတဲ့ ဆံပင်အိတ်နဲ့ ကောင်မလေးလေးပါ။
A man lays on a bench while his dog sits by him .       လူတစ်ယောက်က ခုံပေါ်မှာ လဲနေတယ်၊ သူ့ခွေးက သူ့ဘေးမှာ ထိုင်နေတယ်။
A man lays on the bench to which a white dog is also tied .     လူတစ်ယောက်က ခုံပေါ်မှာ လဲနေတယ်၊ အဲဒီမှာ ခွေးဖြူတစ်ကောင်လည်း ချည်နှောင်ထားတယ်။
a man sleeping on a bench outside with a white and black dog sitting next to him .      လူတစ်ယောက်ဟာ အပြင်ဘက်က ခုံပေါ်မှာ အိပ်နေတယ်၊ သူ့ဘေးမှာ ခွေးဖြူနဲ့ အမည်းတစ်ကောင် ထိုင်နေတယ်။
A shirtless man lies on a park bench with his dog .     အင်္ကျီမပါသူတစ်ယောက်က သူ့ခွေးနဲ့အတူ ပန်းခြံက ခုံပေါ်မှာ လဲနေတယ်။
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/flickr30k$
```

Checking on 28 June 2025 ...  
ဘာသာပြန်တာ မပြီးသေးဘူး။  

```
JSON Payload: {
  "text": "A teenager in striped shorts is leaping into the air on the beach .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ကမ်းခြေမှာ ကမ်းရိုးတန်းနဲ့ ချည်ထားတဲ့ ဆယ်ကျော်သက်တစ်ယောက် လေထဲခုန်နေတယ်။"}
JSON Payload: {
  "text": "The man is playing with a foam football on the beach .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"လူတစ်ယောက်က ကမ်းခြေမှာ ဖော့ဘောလုံးနဲ့ ကစားနေတယ်။"}
JSON Payload: {
  "text": "A man in a kayak is holding his paddle up high .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ကယာကတ်စီးနေတဲ့ အမျိုးသားက သူ့လှော်တံကို မြင့်အောင်ကိုင်ထားတယ်။"}
JSON Payload: {
  "text": "A man is kayaking through rough water .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"လူတစ်ယောက်ဟာ မရေမရာ ရေထဲမှာ ကယာကေ့စီးနေတယ်။"}
```

လက်ရှိ အချိန်ထိ ဘာသာပြန်နေတာက နှစ်သောင်းခုနှစ်ထောင်ကျော် ပဲပြီးသေးတယ်။  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/flickr30k$ wc f2.my
  27706  453113 6513094 f2.my
```

စုစုပေါင်းရှိတာက လေးသောင်းလောက်မို့ ... နောက်ထပ် ၂ ရက်လောက် စောင့်ရဦးမယ်။  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/flickr30k$ wc *
  40460  517166 3395237 captions.txt
  40460  476706 2271132 f2.en
  80920  993872 5666369 total
```

Check again 3:38, 28 June 2025:  

```
JSON Payload: {
  "text": "The young boy opens his arms as waves come crashing near him .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ကောင်လေးက သူ့လက်မောင်းတွေကို ဖွင့်လိုက်ပြီး လှိုင်းတွေ သူ့အနားကို ဝင်လာကြတယ်"}
JSON Payload: {
  "text": "A group of men wearing uniforms with hats gather holding flags .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"တပ္မေတာ္သားမ်ားက တပ္သားအလံမ်ားကို ဆြဲေဆာင္ၿပီး ေခါင္းေဆာင္မ်ားကို စုေပါင္းၾကသည္။"}
JSON Payload: {
  "text": "A VFW group stand at attention at a funeral .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"စစ်မှုထမ်းဟောင်းတွေ အသုဘမှာ ရပ်နေကြတယ်။"}
JSON Payload: {
  "text": "Members of a fraternal organization in a graveyard",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"သုသာန်တစ်ခုမှာ ညီနောင်အဖွဲ့အစည်းရဲ့ အဖွဲ့ဝင်တွေ"}
```

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/flickr30k$ wc f2.my
  31398  513755 7393800 f2.my
```

## Preprocessing

```
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ python3.10 ./extract_filenames.py --input ./captions.txt --output filenames.txt
Converted filenames written to filenames.txt (Total lines: 40460)
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ ls
captions.txt  extract_filenames.py  f2.my  filenames.txt
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ wc filenames.txt
 40460  40460 598115 filenames.txt
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ wc captions.txt
  40460  517166 3395237 captions.txt
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$
```

```
  40460  517166 3395237 captions.txt
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ head filenames.txt
1000268201.jpg
1000268201.jpg
1000268201.jpg
1000268201.jpg
1000268201.jpg
1001773457.jpg
1001773457.jpg
1001773457.jpg
1001773457.jpg
1001773457.jpg
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$
```

```
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ cut -f2 ./f2.my > my
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ head my
ပန်းရောင်ဝတ်စုံဝတ်ထားတဲ့ ကလေးတစ်ယောက်က ဝင်ပေါက်လမ်းမှာ လှေကားတစ်စီးတက်နေတယ်။
သစ်သားအဆောက်အအုံထဲ ဝင်နေတဲ့ မိန်းကလေး။
သစ်သားကစားရုံထဲ တက်လာတဲ့ ကောင်မလေးတစ်ယောက်
သူမရဲ့ ကစားခန်းဆီ လှေကားထစ်တက်နေတဲ့ ကလေးမလေးတစ်ယောက်
ပန်းရောင်ဝတ်စုံနဲ့ သစ်သားအိမ်လေးထဲ ဝင်နေတဲ့ မိန်းကလေးလေး။
ခွေးမည်းနဲ့ ခွေးအစက်တွေ တိုက်နေကြတာ
ခွေးမည်းနဲ့ ခွေးသုံးမျိုး လမ်းပေါ်မှာ ကစားနေကြတယ်။
အနက်ရောင်ခွေးနဲ့ အညိုရောင်အစက်တွေပါတဲ့ အဖြူရောင်ခွေဟာ လမ်းပေါ်မှာ အချင်းချင်း ငေးကြည့်နေကြတယ်။
လမ်းပေါ်မှာ မတူညီတဲ့ ခွေးနှစ်ကောင် အချင်းချင်းကြည့်နေတာပါ။
လမ်းဘေးမှာ ခွေးနှစ်ကောင် အချင်းချင်း ချဉ်းကပ်နေကြတယ်။
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$
```

လောလောဆယ်က NLLB နဲ့ ဘာသာပြန်ပြီးသလောက်ကိုပဲ စမ်းကြည့်ချင်တာမို့လို့ ... 

```
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ head -n 27731 ./filenames.txt > filenames.txt.27k
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ wc filenames.txt.27k
 27731  27731 412905 filenames.txt.27k
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$
```

```
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ paste filenames.txt.27k my > my_captions.txt
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ head my_captions.txt
1000268201.jpg  ပန်းရောင်ဝတ်စုံဝတ်ထားတဲ့ ကလေးတစ်ယောက်က ဝင်ပေါက်လမ်းမှာ လှေကားတစ်စီးတက်နေတယ်။
1000268201.jpg  သစ်သားအဆောက်အအုံထဲ ဝင်နေတဲ့ မိန်းကလေး။
1000268201.jpg  သစ်သားကစားရုံထဲ တက်လာတဲ့ ကောင်မလေးတစ်ယောက်
1000268201.jpg  သူမရဲ့ ကစားခန်းဆီ လှေကားထစ်တက်နေတဲ့ ကလေးမလေးတစ်ယောက်
1000268201.jpg  ပန်းရောင်ဝတ်စုံနဲ့ သစ်သားအိမ်လေးထဲ ဝင်နေတဲ့ မိန်းကလေးလေး။
1001773457.jpg  ခွေးမည်းနဲ့ ခွေးအစက်တွေ တိုက်နေကြတာ
1001773457.jpg  ခွေးမည်းနဲ့ ခွေးသုံးမျိုး လမ်းပေါ်မှာ ကစားနေကြတယ်။
1001773457.jpg  အနက်ရောင်ခွေးနဲ့ အညိုရောင်အစက်တွေပါတဲ့ အဖြူရောင်ခွေဟာ လမ်းပေါ်မှာ အချင်းချင်း ငေးကြည့်နေကြတယ်။
1001773457.jpg  လမ်းပေါ်မှာ မတူညီတဲ့ ခွေးနှစ်ကောင် အချင်းချင်းကြည့်နေတာပါ။
1001773457.jpg  လမ်းဘေးမှာ ခွေးနှစ်ကောင် အချင်းချင်း ချဉ်းကပ်နေကြတယ်။
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$ tail my_captions.txt
3421789737.jpg  ဘဝကို မြတ်နိုးဖို့ ပိုက်လိုင်းပေါ်မှာ ကောင်လေးကို ကိုင်ထားတယ်
3421789737.jpg  လူကြီး အမျိုးသားလက်တစ်ဖက်ကို ရှေ့တန်းမှာ မြင်ရတဲ့ အပြာနဲ့အဝါရောင် ကမ်းလှမ်းကားတစ်စီးကို စီးနေတဲ့ ကောင်လေးတစ်ယောက်
3421789737.jpg  ကလေးငယ်တစ်ယောက်က လှည့်ပတ်ရင်း ရပ်နေတယ်။
3421789737.jpg  ကောင်လေးက အပြာရောင် တြိဂံထဲက ကစားကွင်းပစ္စည်းပေါ်မှာ ရပ်နေတယ်။
3421928157.jpg  အောက်ပိုင်းပြွန်တစ်ခုထဲမှာ လူတစ်စုဟာ ရေစီးကြောင်းတစ်ခုခုကို ဖြတ်သန်းနေကြတယ်။
3421928157.jpg  လူတစ်စုနဲ့အတူ လှေတစ်စင်းဟာ လှိုင်းတိုက်မှုကြောင့် လှည့်ပတ်သွားပါတယ်။
3421928157.jpg  လူတွေဟာ မြစ်ထဲမှာ အပြာရောင် လှေစီးနဲ့ လှဲလှဲနေတာပါ။
3421928157.jpg  အပြာရောင် လေထီးစီးလှေနှစ်စင်းဟာ ရေဖြူပေါ်မှာ တိုက်မိပါတယ်။
3421928157.jpg  မြစ်တစ်စင်းမှာ လှေစီးနှစ်စင်း လှန်သွားတယ်
3422146099.jpg  ခွေးညိုက တုတ်တိုင်ကို ခုန်နေတယ်။
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro$
```

```
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro/token$ python sylbreak.py --input f2 --separator " " > f2.syl
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro/token$ head f2.syl
ပန်း ရောင် ဝတ် စုံ ဝတ် ထား တဲ့ က လေး တစ် ယောက် က ဝင် ပေါက် လမ်း မှာ လှေ ကား တစ် စီး တက် နေ တယ် ။
သစ် သား အ ဆောက် အ အုံ ထဲ ဝင် နေ တဲ့ မိန်း က လေး ။
သစ် သား က စား ရုံ ထဲ တက် လာ တဲ့ ကောင် မ လေး တစ် ယောက်
သူ မ ရဲ့ က စား ခန်း ဆီ လှေ ကား ထစ် တက် နေ တဲ့ က လေး မ လေး တစ် ယောက်
ပန်း ရောင် ဝတ် စုံ နဲ့ သစ် သား အိမ် လေး ထဲ ဝင် နေ တဲ့ မိန်း က လေး လေး ။
ခွေး မည်း နဲ့ ခွေး အ စက် တွေ တိုက် နေ ကြ တာ
ခွေး မည်း နဲ့ ခွေး သုံး မျိုး လမ်း ပေါ် မှာ က စား နေ ကြ တယ် ။
အ နက် ရောင် ခွေး နဲ့ အ ညို ရောင် အ စက် တွေ ပါ တဲ့ အ ဖြူ ရောင် ခွေ ဟာ လမ်း ပေါ် မှာ အ ချင်း ချင်း ငေး ကြ ည့် နေ ကြ တယ် ။
လမ်း ပေါ် မှာ မ တူ ညီ တဲ့ ခွေး နှစ် ကောင် အ ချင်း ချင်း ကြ ည့် နေ တာ ပါ ။
လမ်း ဘေး မှာ ခွေး နှစ် ကောင် အ ချင်း ချင်း ချဉ်း ကပ် နေ ကြ တယ် ။
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro/token$
```

```
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro/token$ cut -f1 ./my_captions.txt > f1
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro/token$ paste f1 f2.syl > my_captions_syl.txt
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro/token$ head my_captions_syl.txt
1000268201.jpg  ပန်း ရောင် ဝတ် စုံ ဝတ် ထား တဲ့ က လေး တစ် ယောက် က ဝင် ပေါက် လမ်း မှာ လှေ ကား တစ် စီး တက် နေ တယ် ။
1000268201.jpg  သစ် သား အ ဆောက် အ အုံ ထဲ ဝင် နေ တဲ့ မိန်း က လေး ။
1000268201.jpg  သစ် သား က စား ရုံ ထဲ တက် လာ တဲ့ ကောင် မ လေး တစ် ယောက်
1000268201.jpg  သူ မ ရဲ့ က စား ခန်း ဆီ လှေ ကား ထစ် တက် နေ တဲ့ က လေး မ လေး တစ် ယောက်
1000268201.jpg  ပန်း ရောင် ဝတ် စုံ နဲ့ သစ် သား အိမ် လေး ထဲ ဝင် နေ တဲ့ မိန်း က လေး လေး ။
1001773457.jpg  ခွေး မည်း နဲ့ ခွေး အ စက် တွေ တိုက် နေ ကြ တာ
1001773457.jpg  ခွေး မည်း နဲ့ ခွေး သုံး မျိုး လမ်း ပေါ် မှာ က စား နေ ကြ တယ် ။
1001773457.jpg  အ နက် ရောင် ခွေး နဲ့ အ ညို ရောင် အ စက် တွေ ပါ တဲ့ အ ဖြူ ရောင် ခွေ ဟာ လမ်း ပေါ် မှာ အ ချင်း ချင်း ငေး ကြ ည့် နေ ကြ တယ် ။
1001773457.jpg  လမ်း ပေါ် မှာ မ တူ ညီ တဲ့ ခွေး နှစ် ကောင် အ ချင်း ချင်း ကြ ည့် နေ တာ ပါ ။
1001773457.jpg  လမ်း ဘေး မှာ ခွေး နှစ် ကောင် အ ချင်း ချင်း ချဉ်း ကပ် နေ ကြ တယ် ။
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro/token$
```

```
(base) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro/token$ cp my_captions_syl.txt ../../../data/
```

လက်ရှိ Run နေတဲ့ ep100 (အင်္ဂလိပ်စာ captions နဲ့) ပြီးသွားရင်၊ မြန်မာ လေဘယ်နဲ့ စမ်း run ကြည်မယ်။  

## Translation for Flickr30k Finished  

```
}
API Response: {"result":"လူတစ်ယောက်ဟာ လေထဲမှာ မြင့်မားစွာ ကျောက်တန်းတက်နေတယ်။"}
JSON Payload: {
  "text": "A person in a red shirt climbing up a rock face covered in assist handles .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"အနီရောင် အင်္ကျီဝတ်ပြီး အကူအညီပေးတဲ့ လက်ကိုင်တွေနဲ့ ဖုံးအုပ်ထားတဲ့ ကျောက်တုံးတစ်လုံးပေါ် တက်နေတဲ့လူတစ်ယောက်။"}
JSON Payload: {
  "text": "A rock climber in a red shirt .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"အနီရောင် အင်္ကျီဝတ်ထားတဲ့ ကျောက်တန်းတက်သမားတစ်ယောက်။"}
JSON Payload: {
  "text": "A rock climber practices on a rock climbing wall .",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ကျောက်တက်သမားဟာ ကျောက်တက်နံရံပေါ်မှာ လေ့ကျင့်တယ်။"}

real    6176m41.985s
user    93m44.679s
sys     7m5.231s
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb$
```

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/flickr30k$ wc f2.my
  40460  662630 9539480 f2.my
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/flickr30k$
```

## Preprocessing Again for All Data  

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$ head f2.my
A child in a pink dress is climbing up a set of stairs in an entry way .        ပန်းရောင်ဝတ်စစုံဝတ်ထားတတဲ့ ကလေးတစ်ယောက်က ဝင်ပေါက်လမ်းမှာ လှေကားတစ်စီးတက်နေတယ်။
A girl going into a wooden building .   သစ်သားအဆောက်အအအုံထဲ ဝင်နေတတဲ့ မိန်းကလေး။
A little girl climbing into a wooden playhouse .        သစ်သားကစားရရုံထဲ တက်လာတတဲ့ ကောင်မလေးတစ်ယောက်
A little girl climbing the stairs to her playhouse .    သူမရရဲ့ ကစားခန်းဆီ လှေကားထစ်တက်နေတတဲ့ ကလေးမလေးတစ်ယောက်
A little girl in a pink dress going into a wooden cabin .       ပန်းရောင်ဝတ်စစုံနနဲ့ သစ်သားအိမ်လေးထဲ ဝင်နေတတဲ့ မိန်းကလေးလေး။
A black dog and a spotted dog are fighting      ခွေးမည်းနနဲ့ ခွေးအစက်တွေ တတိုက်နေကြတာ
A black dog and a tri-colored dog playing with each other on the road . ခွေးမည်းနနဲ့ ခွေးသသုံးမျျိုး လမ်းပေါ်မှာ ကစားနေကြတယ်။
A black dog and a white dog with brown spots are staring at each other in the street .  အနက်ရောင်ခွေးနနဲ့ အညညိုရောင်အစက်တွေပါတတဲ့ အဖြူရောင်ခွေဟာ လမ်းပေါ်မှာ အချင်းချင်း ငေးကြည့်နေကြတယ်။
Two dogs of different breeds looking at each other on the road .        လမ်းပေါ်မှာ မတူညီတတဲ့ ခွေးနှစ်ကောင် အချင်းချင်းကြည့်နေတာပါ။
Two dogs on pavement moving toward each other . လမ်းဘေးမှာ ခွေးနှစ်ကောင် အချင်းချင်း ချဉ်းကပ်နေကြတယ်။
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$ cut -f2 ./f2.my > my
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$ head my
ပန်းရောင်ဝတ်စစုံဝတ်ထားတတဲ့ ကလေးတစ်ယောက်က ဝင်ပေါက်လမ်းမှာ လှေကားတစ်စီးတက်နေတယ်။
သစ်သားအဆောက်အအအုံထဲ ဝင်နေတတဲ့ မိန်းကလေး။
သစ်သားကစားရရုံထဲ တက်လာတတဲ့ ကောင်မလေးတစ်ယောက်
သူမရရဲ့ ကစားခန်းဆီ လှေကားထစ်တက်နေတတဲ့ ကလေးမလေးတစ်ယောက်
ပန်းရောင်ဝတ်စစုံနနဲ့ သစ်သားအိမ်လေးထဲ ဝင်နေတတဲ့ မိန်းကလေးလေး။
ခွေးမည်းနနဲ့ ခွေးအစက်တွေ တတိုက်နေကြတာ
ခွေးမည်းနနဲ့ ခွေးသသုံးမျျိုး လမ်းပေါ်မှာ ကစားနေကြတယ်။
အနက်ရောင်ခွေးနနဲ့ အညညိုရောင်အစက်တွေပါတတဲ့ အဖြူရောင်ခွေဟာ လမ်းပေါ်မှာ အချင်းချင်း ငေးကြည့်နေကြတယ်။
လမ်းပေါ်မှာ မတူညီတတဲ့ ခွေးနှစ်ကောင် အချင်းချင်းကြည့်နေတာပါ။
လမ်းဘေးမှာ ခွေးနှစ်ကောင် အချင်းချင်း ချဉ်းကပ်နေကြတယ်။
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$ head captions.txt
1000268201_693b08cb0e.jpg#0     A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg#1     A girl going into a wooden building .
1000268201_693b08cb0e.jpg#2     A little girl climbing into a wooden playhouse .
1000268201_693b08cb0e.jpg#3     A little girl climbing the stairs to her playhouse .
1000268201_693b08cb0e.jpg#4     A little girl in a pink dress going into a wooden cabin .
1001773457_577c3a7d70.jpg#0     A black dog and a spotted dog are fighting
1001773457_577c3a7d70.jpg#1     A black dog and a tri-colored dog playing with each other on the road .
1001773457_577c3a7d70.jpg#2     A black dog and a white dog with brown spots are staring at each other in the street .
1001773457_577c3a7d70.jpg#3     Two dogs of different breeds looking at each other on the road .
1001773457_577c3a7d70.jpg#4     Two dogs on pavement moving toward each other .
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$
```

filename ကို ဆွဲထုတ်မယ်။  

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$ python3.10 ./extract_filenames.py --input ./captions.txt --output filenames.txt
Converted filenames written to filenames.txt (Total lines: 40460)
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$ head ./filenames.txt
1000268201.jpg
1000268201.jpg
1000268201.jpg
1000268201.jpg
1000268201.jpg
1001773457.jpg
1001773457.jpg
1001773457.jpg
1001773457.jpg
1001773457.jpg
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$
```

caption ဖိုင် မြန်မာဘာသာပြန် flickr30k အကုန်အတွက် ရပြီ။  

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$ paste filenames.txt my > my_captions.txt
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$ head my_captions.txt
1000268201.jpg  ပန်းရောင်ဝတ်စစုံဝတ်ထားတတဲ့ ကလေးတစ်ယောက်က ဝင်ပေါက်လမ်းမှာ လှေကားတစ်စီးတက်နေတယ်။
1000268201.jpg  သစ်သားအဆောက်အအအုံထဲ ဝင်နေတတဲ့ မိန်းကလေး။
1000268201.jpg  သစ်သားကစားရရုံထဲ တက်လာတတဲ့ ကောင်မလေးတစ်ယောက်
1000268201.jpg  သူမရရဲ့ ကစားခန်းဆီ လှေကားထစ်တက်နေတတဲ့ ကလေးမလေးတစ်ယောက်
1000268201.jpg  ပန်းရောင်ဝတ်စစုံနနဲ့ သစ်သားအိမ်လေးထဲ ဝင်နေတတဲ့ မိန်းကလေးလေး။
1001773457.jpg  ခွေးမည်းနနဲ့ ခွေးအစက်တွေ တတိုက်နေကြတာ
1001773457.jpg  ခွေးမည်းနနဲ့ ခွေးသသုံးမျျိုး လမ်းပေါ်မှာ ကစားနေကြတယ်။
1001773457.jpg  အနက်ရောင်ခွေးနနဲ့ အညညိုရောင်အစက်တွေပါတတဲ့ အဖြူရောင်ခွေဟာ လမ်းပေါ်မှာ အချင်းချင်း ငေးကြည့်နေကြတယ်။
1001773457.jpg  လမ်းပေါ်မှာ မတူညီတတဲ့ ခွေးနှစ်ကောင် အချင်းချင်းကြည့်နေတာပါ။
1001773457.jpg  လမ်းဘေးမှာ ခွေးနှစ်ကောင် အချင်းချင်း ချဉ်းကပ်နေကြတယ်။
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2$ wc my_captions.txt
  40460  226384 7866463 my_captions.txt
```

phrase level တွဲထားတဲ့ ဖိုင် ရပြီ။  

## Syllable Breaking

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2/syl$ python3.10 ./sylbreak.py --input ../my --separator " " --output my.syl
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2/syl$ head my.syl
ပန်း ရောင် ဝတ် စစုံ ဝတ် ထား တတဲ့ က လေး တစ် ယောက် က ဝင် ပေါက် လမ်း မှာ လှေ ကား တစ် စီး တက် နေ တယ် ။
သစ် သား အ ဆောက် အ အအုံ ထဲ ဝင် နေ တတဲ့ မိန်း က လေး ။
သစ် သား က စား ရရုံ ထဲ တက် လာ တတဲ့ ကောင် မ လေး တစ် ယောက်
သူ မ ရရဲ့ က စား ခန်း ဆီ လှေ ကား ထစ် တက် နေ တတဲ့ က လေး မ လေး တစ် ယောက်
ပန်း ရောင် ဝတ် စစုံ နနဲ့ သစ် သား အိမ် လေး ထဲ ဝင် နေ တတဲ့ မိန်း က လေး လေး ။
ခွေး မည်း နနဲ့ ခွေး အ စက် တွေ တတိုက် နေ ကြ တာ
ခွေး မည်း နနဲ့ ခွေး သသုံး မျျိုး လမ်း ပေါ် မှာ က စား နေ ကြ တယ် ။
အ နက် ရောင် ခွေး နနဲ့ အ ညညို ရောင် အ စက် တွေ ပါ တတဲ့ အ ဖြူ ရောင် ခွေ ဟာ လမ်း ပေါ် မှာ အ ချင်း ချင်း ငေး ကြ ည့် နေ ကြ တယ် ။
လမ်း ပေါ် မှာ မ တူ ညီ တတဲ့ ခွေး နှစ် ကောင် အ ချင်း ချင်း ကြ ည့် နေ တာ ပါ ။
လမ်း ဘေး မှာ ခွေး နှစ် ကောင် အ ချင်း ချင်း ချဉ်း ကပ် နေ ကြ တယ် ။
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2/syl$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2/syl$ paste ../filenames.txt ./my.syl > my_captions.syl
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2/syl$ head my_captions.syl
1000268201.jpg  ပန်း ရောင် ဝတ် စစုံ ဝတ် ထား တတဲ့ က လေး တစ် ယောက် က ဝင် ပေါက် လမ်း မှာ လှေ ကား တစ် စီး တက် နေ တယ် ။
1000268201.jpg  သစ် သား အ ဆောက် အ အအုံ ထဲ ဝင် နေ တတဲ့ မိန်း က လေး ။
1000268201.jpg  သစ် သား က စား ရရုံ ထဲ တက် လာ တတဲ့ ကောင် မ လေး တစ် ယောက်
1000268201.jpg  သူ မ ရရဲ့ က စား ခန်း ဆီ လှေ ကား ထစ် တက် နေ တတဲ့ က လေး မ လေး တစ် ယောက်
1000268201.jpg  ပန်း ရောင် ဝတ် စစုံ နနဲ့ သစ် သား အိမ် လေး ထဲ ဝင် နေ တတဲ့ မိန်း က လေး လေး ။
1001773457.jpg  ခွေး မည်း နနဲ့ ခွေး အ စက် တွေ တတိုက် နေ ကြ တာ
1001773457.jpg  ခွေး မည်း နနဲ့ ခွေး သသုံး မျျိုး လမ်း ပေါ် မှာ က စား နေ ကြ တယ် ။
1001773457.jpg  အ နက် ရောင် ခွေး နနဲ့ အ ညညို ရောင် အ စက် တွေ ပါ တတဲ့ အ ဖြူ ရောင် ခွေ ဟာ လမ်း ပေါ် မှာ အ ချင်း ချင်း ငေး ကြ ည့် နေ ကြ တယ် ။
1001773457.jpg  လမ်း ပေါ် မှာ မ တူ ညီ တတဲ့ ခွေး နှစ် ကောင် အ ချင်း ချင်း ကြ ည့် နေ တာ ပါ ။
1001773457.jpg  လမ်း ဘေး မှာ ခွေး နှစ် ကောင် အ ချင်း ချင်း ချဉ်း ကပ် နေ ကြ တယ် ။
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my/prepro2/syl$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/data/flickr30k_images$ rm my_captions.txt
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/data/flickr30k_images$ cp ../my_captions.syl .
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/data/flickr30k_images$ mv my_captions.syl my_captions.txt
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/data/flickr30k_images$ wc my_captions.txt
  40460  858582 8498780 my_captions.txt
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/data/flickr30k_images$
```

## Image2Text or Captioning for MyanmarFlickr30K  

Epoch 100 နဲ့ Run မယ်။  
ပြီးရင် လက်ရှိ ဗားရှင်းကို လေ့လာလို့ရအောင် ရှဲပေးထားလိုက်မယ်။  

(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ cat ./train.sh  

```bash
#!/bin/bash

# For Myanmar Flickr30k (Testing)
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./my_ep50 --epochs 50 \
#--gpu | tee my_train_ep50.log

# For English Flickr30K
#time python3.10 captioner.py --data_dir ./data/flickr30k_images --caption_file captions.txt --output_dir ./ep100 --epochs 100 \
#--gpu | tee train_100.log

# For Myanmar Flickr30k (Complete version)
time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./myFlickr30k_ep100 --epochs 100 \
--gpu | tee train_100_myFlickr30k.log
```

start training ...  

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ ./train.sh
💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
You are using a CUDA device ('NVIDIA GeForce RTX 3090 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
2025-06-30 05:38:47.846204: I tensorflow/core/util/util.cc:169] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-06-30 05:38:47.848472: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/lib:/usr/local/lib:
2025-06-30 05:38:47.848483: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

   | Name            | Type                  | Params | Mode
-------------------------------------------------------------------
0  | val_bleu        | CorpusBleu            | 0      | train
1  | test_bleu       | CorpusBleu            | 0      | train
2  | rouge           | ROUGEScore            | 0      | train
3  | chrf            | CHRFScore             | 0      | train
4  | train_bleu      | CorpusBleu            | 0      | train
5  | train_chrf      | CHRFScore             | 0      | train
6  | val_chrf        | CHRFScore             | 0      | train
7  | image_extractor | ImageFeatureExtractor | 25.1 M | train
8  | word_embedder   | WordEmbedder          | 1.1 M  | train
9  | decoder         | LSTM                  | 23.1 M | train
10 | fc_scorer       | ParallelFCScorer      | 2.1 M  | train
-------------------------------------------------------------------
28.4 M    Trainable params
23.0 M    Non-trainable params
51.4 M    Total params
205.436   Total estimated model params size (MB)
19        Modules in train mode
148       Modules in eval mode
Successfully loaded 40459 images with captions
Starting training...
Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]
...
...
Epoch 1:  86%|████████▌ | 81/94 [00:16<00:02,  4.99it/s, v_num=0, train_loss_step=3.910, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  87%|████████▋ | 82/94 [00:16<00:02,  5.04it/s, v_num=0, train_loss_step=3.910, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  87%|████████▋ | 82/94 [00:16<00:02,  5.03it/s, v_num=0, train_loss_step=3.500, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  88%|████████▊ | 83/94 [00:16<00:02,  5.08it/s, v_num=0, train_loss_step=3.500, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  88%|████████▊ | 83/94 [00:16<00:02,  5.07it/s, v_num=0, train_loss_step=3.850, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  89%|████████▉ | 84/94 [00:16<00:01,  5.12it/s, v_num=0, train_loss_step=3.850, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  89%|████████▉ | 84/94 [00:16<00:01,  5.12it/s, v_num=0, train_loss_step=3.670, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  90%|█████████ | 85/94 [00:16<00:01,  5.16it/s, v_num=0, train_loss_step=3.670, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  90%|█████████ | 85/94 [00:16<00:01,  5.16it/s, v_num=0, train_loss_step=3.650, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  91%|█████████▏| 86/94 [00:16<00:01,  5.20it/s, v_num=0, train_loss_step=3.650, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  91%|█████████▏| 86/94 [00:16<00:01,  5.20it/s, v_num=0, train_loss_step=3.930, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  93%|█████████▎| 87/94 [00:16<00:01,  5.24it/s, v_num=0, train_loss_step=3.930, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  93%|█████████▎| 87/94 [00:16<00:01,  5.24it/s, v_num=0, train_loss_step=3.480, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  94%|█████████▎| 88/94 [00:16<00:01,  5.28it/s, v_num=0, train_loss_step=3.480, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  94%|█████████▎| 88/94 [00:16<00:01,  5.28it/s, v_num=0, train_loss_step=3.600, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  95%|█████████▍| 89/94 [00:16<00:00,  5.32it/s, v_num=0, train_loss_step=3.600, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  95%|█████████▍| 89/94 [00:16<00:00,  5.32it/s, v_num=0, train_loss_step=3.770, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  96%|█████████▌| 90/94 [00:16<00:00,  5.36it/s, v_num=0, train_loss_step=3.770, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  96%|█████████▌| 90/94 [00:16<00:00,  5.35it/s, v_num=0, train_loss_step=3.780, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  97%|█████████▋| 91/94 [00:16<00:00,  5.40it/s, v_num=0, train_loss_step=3.780, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  97%|█████████▋| 91/94 [00:16<00:00,  5.39it/s, v_num=0, train_loss_step=3.770, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  98%|█████████▊| 92/94 [00:16<00:00,  5.43it/s, v_num=0, train_loss_step=3.770, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  98%|█████████▊| 92/94 [00:16<00:00,  5.43it/s, v_num=0, train_loss_step=3.700, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  99%|█████████▉| 93/94 [00:16<00:00,  5.47it/s, v_num=0, train_loss_step=3.700, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1:  99%|█████████▉| 93/94 [00:17<00:00,  5.47it/s, v_num=0, train_loss_step=3.840, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1: 100%|██████████| 94/94 [00:17<00:00,  5.51it/s, v_num=0, train_loss_step=3.840, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=Epoch 1: 100%|██████████| 94/94 [00:17<00:00,  5.51it/s, v_num=0, train_loss_step=3.720, val_loss=4.140, val_bleu=0.00763, val_chrf=0.0386, train_loss_epoch=4.820]                                                         Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
                                                                       Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])t/s]
Projection input dimension: 2048
Projection output dimension: 1024
                                                                       Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])t/s]
Projection input dimension: 2048
Projection output dimension: 1024

Validation DataLoader 0:  19%|█▉        | 3/16 [00:31<02:15,  0.10it/s]
...
...
...
Projection output dimension: 1024
Testing DataLoader 0:  81%|████████▏ | 13/16 [01:51<00:25,  0.12it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0:  88%|████████▊ | 14/16 [02:00<00:17,  0.12it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0:  94%|█████████▍| 15/16 [02:08<00:08,  0.12it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0: 100%|██████████| 16/16 [02:17<00:00,  0.12it/s]
TESTING COMPLETE - SUMMARY METRICS:
==================================================
BLEU-1: 0.4379
BLEU-2: 0.2625
BLEU-4: 0.1216
METEOR: 0.3092
ROUGE-L: 0.0013
CHRF++: 0.0296
Semantic_Similarity: 0.1896
Keyword_Overlap: 0.4659
Composite_Score: 0.1896
CIDEr: 0.0000
==================================================

Detailed results saved to: ./myFlickr30k_ep100/detailed_predictions.json
Human-readable results saved to: ./myFlickr30k_ep100/predictions.txt
Testing DataLoader 0: 100%|██████████| 16/16 [02:45<00:00,  0.10it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_loss         │     1.992755651473999     │
└───────────────────────────┴───────────────────────────┘

real    314m2.529s
user    348m21.562s
sys     4m57.470s
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ tree ./myFlickr30k_ep100/
./myFlickr30k_ep100/
├── detailed_predictions.json
├── final_model.ckpt
├── lightning_logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=99-step=9400.ckpt
│       ├── events.out.tfevents.1751236728.lst-hpc3090.1935589.0
│       ├── events.out.tfevents.1751255400.lst-hpc3090.1935589.1
│       └── hparams.yaml
├── predictions.txt
├── test_loss.png
└── train_val_loss.png

4 directories, 9 files
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ du -h ./myFlickr30k_ep100/
413M    ./myFlickr30k_ep100/lightning_logs/version_0/checkpoints
413M    ./myFlickr30k_ep100/lightning_logs/version_0
413M    ./myFlickr30k_ep100/lightning_logs
831M    ./myFlickr30k_ep100/
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

## Code Updating/Debugging with vgg16

Training with vgg16 ...  

time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./myFlickr30k_vgg_ep100 --epochs 100 \
--gpu --encoder vgg16 | tee train_100_myFlickr30k_vgg16.log  

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ ./train.sh
...
...
...
Projection output dimension: 1024
Testing DataLoader 0:  81%|████████▏ | 13/16 [01:52<00:25,  0.12it/s]Encoder output features before pooling: torch.Size([64, 512, 7, 7])
Encoder output features after pooling: torch.Size([64, 512, 1, 1])
Projection input dimension: 512
Projection output dimension: 1024
Testing DataLoader 0:  88%|████████▊ | 14/16 [02:00<00:17,  0.12it/s]Encoder output features before pooling: torch.Size([64, 512, 7, 7])
Encoder output features after pooling: torch.Size([64, 512, 1, 1])
Projection input dimension: 512
Projection output dimension: 1024
Testing DataLoader 0:  94%|█████████▍| 15/16 [02:09<00:08,  0.12it/s]Encoder output features before pooling: torch.Size([64, 512, 7, 7])
Encoder output features after pooling: torch.Size([64, 512, 1, 1])
Projection input dimension: 512
Projection output dimension: 1024
Testing DataLoader 0: 100%|██████████| 16/16 [02:18<00:00,  0.12it/s]
TESTING COMPLETE - SUMMARY METRICS:
==================================================
BLEU-1: 0.4429
BLEU-2: 0.2712
BLEU-4: 0.1265
METEOR: 0.3096
ROUGE-L: 0.0017
CHRF++: 0.0298
Semantic_Similarity: 0.1896
Keyword_Overlap: 0.4727
Composite_Score: 0.1896
CIDEr: 0.0000
==================================================

Detailed results saved to: ./myFlickr30k_vgg_ep100/detailed_predictions.json
Human-readable results saved to: ./myFlickr30k_vgg_ep100/predictions.txt
Testing DataLoader 0: 100%|██████████| 16/16 [02:45<00:00,  0.10it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_loss         │    1.9882290363311768     │
└───────────────────────────┴───────────────────────────┘

real    319m58.576s
user    353m47.744s
sys     4m50.484s
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ tree ./myFlickr30k_vgg_ep100/
./myFlickr30k_vgg_ep100/
├── detailed_predictions.json
├── final_model.ckpt
├── lightning_logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=99-step=9400.ckpt
│       ├── events.out.tfevents.1751299502.lst-hpc3090.2045453.0
│       ├── events.out.tfevents.1751318529.lst-hpc3090.2045453.1
│       └── hparams.yaml
├── predictions.txt
├── test_loss.png
└── train_val_loss.png

4 directories, 9 files
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

## Training 100 Epoch with mobilenet_v2

(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ cat ./train.sh  

```
#!/bin/bash

# For Myanmar Flickr30k (Testing)
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./my_ep50 --epochs 50 \
#--gpu | tee my_train_ep50.log

# For English Flickr30K
#time python3.10 captioner.py --data_dir ./data/flickr30k_images --caption_file captions.txt --output_dir ./ep100 --epochs 100 \
#--gpu | tee train_100.log

# For Myanmar Flickr30k (Complete version) with default resnext50
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./myFlickr30k_ep100 --epochs 100 \
#--gpu | tee train_100_myFlickr30k.log

        # Map user-friendly names to actual model names
#        model_map = {
#            "resnet50": "resnet50",
#            "resnet101": "resnet101",
#            "resnet152": "resnet152",
#            "mobilenetv2": "mobilenet_v2",
#            "vgg16": "vgg16",
#            "resnext50": "resnext50_32x4d",
#            "resnext101": "resnext101_32x8d"
#        }

# For Myanmar Flickr30k (Complete version) with VGG16
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./myFlickr30k_vgg_ep100 --epochs 100 \
#--gpu --encoder vgg16 | tee train_100_myFlickr30k_vgg16.log

# For Myanmar Flickr30k (Complete version) with mobilenetv2
time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./myFlickr30k_mobile_ep100 --epochs 100 \
--gpu --encoder mobilenetv2 | tee train_100_myFlickr30k_mobile.log
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ ./train.sh
...
...
...
Projection output dimension: 1024
Testing DataLoader 0:  81%|████████▏ | 13/16 [01:51<00:25,  0.12it/s]Encoder output features before pooling: torch.Size([64, 1280, 7, 7])
Encoder output features after pooling: torch.Size([64, 1280, 1, 1])
Projection input dimension: 1280
Projection output dimension: 1024
Testing DataLoader 0:  88%|████████▊ | 14/16 [02:00<00:17,  0.12it/s]Encoder output features before pooling: torch.Size([64, 1280, 7, 7])
Encoder output features after pooling: torch.Size([64, 1280, 1, 1])
Projection input dimension: 1280
Projection output dimension: 1024
Testing DataLoader 0:  94%|█████████▍| 15/16 [02:08<00:08,  0.12it/s]Encoder output features before pooling: torch.Size([64, 1280, 7, 7])
Encoder output features after pooling: torch.Size([64, 1280, 1, 1])
Projection input dimension: 1280
Projection output dimension: 1024
Testing DataLoader 0: 100%|██████████| 16/16 [02:17<00:00,  0.12it/s]
TESTING COMPLETE - SUMMARY METRICS:
==================================================
BLEU-1: 0.4334
BLEU-2: 0.2534
BLEU-4: 0.1152
METEOR: 0.2941
ROUGE-L: 0.0013
CHRF++: 0.0296
Semantic_Similarity: 0.1841
Keyword_Overlap: 0.4563
Composite_Score: 0.1841
CIDEr: 0.0000
==================================================

Detailed results saved to: ./myFlickr30k_mobile_ep100/detailed_predictions.json
Human-readable results saved to: ./myFlickr30k_mobile_ep100/predictions.txt
Testing DataLoader 0: 100%|██████████| 16/16 [02:45<00:00,  0.10it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_loss         │     1.998742938041687     │
└───────────────────────────┴───────────────────────────┘

real    313m9.788s
user    344m49.750s
sys     4m55.946s
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ tree ./myFlickr30k_mobile_ep100/
./myFlickr30k_mobile_ep100/
├── detailed_predictions.json
├── final_model.ckpt
├── lightning_logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=99-step=9400.ckpt
│       ├── events.out.tfevents.1751321449.lst-hpc3090.2086353.0
│       ├── events.out.tfevents.1751340067.lst-hpc3090.2086353.1
│       └── hparams.yaml
├── predictions.txt
├── test_loss.png
└── train_val_loss.png

4 directories, 9 files
```

## Train 100 epoch with resnext101 

time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./myFlickr30k_resnext101_ep100 --epochs 100 \
--gpu --encoder resnext101 | tee train_100_myFlickr30k_resnext101.log    

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ ./train.sh
Downloading: "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth" to /home/ye/.cache/torch/hub/checkpoints/resnext101_32x8d-8ba56ff5.pth
Successfully loaded 40459 images with captions
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 340M/340M [00:03<00:00, 98.1MB/s]
💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
You are using a CUDA device ('NVIDIA GeForce RTX 3090 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
2025-07-01 21:55:25.577711: I tensorflow/core/util/util.cc:169] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-07-01 21:55:25.579947: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/lib:/usr/local/lib:
2025-07-01 21:55:25.579959: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

   | Name            | Type                  | Params | Mode
-------------------------------------------------------------------
0  | val_bleu        | CorpusBleu            | 0      | train
1  | test_bleu       | CorpusBleu            | 0      | train
2  | rouge           | ROUGEScore            | 0      | train
3  | chrf            | CHRFScore             | 0      | train
4  | train_bleu      | CorpusBleu            | 0      | train
5  | train_chrf      | CHRFScore             | 0      | train
6  | val_chrf        | CHRFScore             | 0      | train
7  | image_extractor | ImageFeatureExtractor | 88.8 M | train
8  | word_embedder   | WordEmbedder          | 1.1 M  | train
9  | decoder         | LSTM                  | 23.1 M | train
10 | fc_scorer       | ParallelFCScorer      | 2.1 M  | train
-------------------------------------------------------------------
28.4 M    Trainable params
86.7 M    Non-trainable params
115 M     Total params
460.486   Total estimated model params size (MB)
19        Modules in train mode
284       Modules in eval mode
Starting training...
Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Sanity Checking DataLoader 0:  50%|█████     | 1/2 [00:11<00:11,  0.09it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Epoch 0:   0%|          | 0/94 [00:00<?, ?it/s] Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Epoch 0: 100%|██████████| 94/94 [00:24<00:00,  3.82it/s, v_num=Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024

Validation DataLoader 0:   6%|▋         | 1/16 [00:10<02:40,  0.09it/s]
...
...
...
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0:  94%|█████████▍| 15/16 [02:11<00:08,  0.11it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0: 100%|██████████| 16/16 [02:20<00:00,  0.11it/s]
TESTING COMPLETE - SUMMARY METRICS:
==================================================
BLEU-1: 0.4310
BLEU-2: 0.2634
BLEU-4: 0.1240
METEOR: 0.3022
ROUGE-L: 0.0023
CHRF++: 0.0293
Semantic_Similarity: 0.1846
Keyword_Overlap: 0.4593
Composite_Score: 0.1846
CIDEr: 0.0000
==================================================

Detailed results saved to: ./myFlickr30k_resnext101_ep100/detailed_predictions.json
Human-readable results saved to: ./myFlickr30k_resnext101_ep100/predictions.txt
Testing DataLoader 0: 100%|██████████| 16/16 [02:48<00:00,  0.10it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_loss         │    1.9963449239730835     │
└───────────────────────────┴───────────────────────────┘

real    332m48.373s
user    365m48.287s
sys     5m11.063s
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

check outputs ... 

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ tree ./myFlickr30k_resnext101_ep100/
./myFlickr30k_resnext101_ep100/
├── detailed_predictions.json
├── final_model.ckpt
├── lightning_logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=99-step=9400.ckpt
│       ├── events.out.tfevents.1751381726.lst-hpc3090.2190785.0
│       ├── events.out.tfevents.1751401516.lst-hpc3090.2190785.1
│       └── hparams.yaml
├── predictions.txt
├── test_loss.png
└── train_val_loss.png

4 directories, 9 files
```

Current Folders:  

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ ls
bk               myFlickr30k_ep100             prepro                     train_100_myFlickr30k_mobile.log
captioner_my.py  myFlickr30k_mobile_ep100      prepro2                    train_100_myFlickr30k_resnext101.log
my_ep1           myFlickr30k_resnext101_ep100  resume_50.log              train_100_myFlickr30k_vgg16.log
my_ep100         myFlickr30k_vgg_ep100         resume_train.sh            train.sh
my_ep200         my_train_ep1.log              tmp.log
my_ep50          my_train_ep50.log             train_100_myFlickr30k.log
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

Updated the shell script:  

```bash
# For Myanmar Flickr30k (Complete version) with resnet50
time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt \
--output_dir ./myFlickr30k_resnet50 --gpu --encoder resnet50 | tee train_100_myFlickr30k_resnet50.log
```

start training with resnet50  

```
Testing DataLoader 0:  88%|████████▊ | 14/16 [02:01<00:17,  0.12it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0:  94%|█████████▍| 15/16 [02:10<00:08,  0.12it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0: 100%|██████████| 16/16 [02:18<00:00,  0.12it/s]
TESTING COMPLETE - SUMMARY METRICS:
==================================================
BLEU-1: 0.4409
BLEU-2: 0.2498
BLEU-4: 0.1026
METEOR: 0.2780
ROUGE-L: 0.0000
CHRF++: 0.0299
Semantic_Similarity: 0.1851
Keyword_Overlap: 0.4629
Composite_Score: 0.1851
CIDEr: 0.0000
==================================================

Detailed results saved to: ./myFlickr30k_resnet50/detailed_predictions.json
Human-readable results saved to: ./myFlickr30k_resnet50/predictions.txt
Testing DataLoader 0: 100%|██████████| 16/16 [02:46<00:00,  0.10it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_loss         │    2.3281002044677734     │
└───────────────────────────┴───────────────────────────┘

real    65m34.667s
user    72m33.891s
sys     1m5.057s
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ tree  ./myFlickr30k_resnet50/
./myFlickr30k_resnet50/
├── detailed_predictions.json
├── final_model.ckpt
├── lightning_logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=19-step=1880.ckpt
│       ├── events.out.tfevents.1751427895.lst-hpc3090.2272456.0
│       ├── events.out.tfevents.1751431656.lst-hpc3090.2272456.1
│       └── hparams.yaml
├── predictions.txt
├── test_loss.png
└── train_val_loss.png

4 directories, 9 files
```

## Train resnet101  

# For Myanmar Flickr30k (Complete version) with resnet101
time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt \
--output_dir ./myFlickr30k_resnet101 --gpu --encoder resnet101 | tee train_100_myFlickr30k_resnet101.log
  
```
Testing DataLoader 0:  88%|████████▊ | 14/16 [02:01<00:17,  0.12it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0:  94%|█████████▍| 15/16 [02:09<00:08,  0.12it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0: 100%|██████████| 16/16 [02:18<00:00,  0.12it/s]
TESTING COMPLETE - SUMMARY METRICS:
==================================================
BLEU-1: 0.4387
BLEU-2: 0.2644
BLEU-4: 0.1073
METEOR: 0.2781
ROUGE-L: 0.0000
CHRF++: 0.0309
Semantic_Similarity: 0.1836
Keyword_Overlap: 0.4871
Composite_Score: 0.1836
CIDEr: 0.0000
==================================================

Detailed results saved to: ./myFlickr30k_resnet101/detailed_predictions.json
Human-readable results saved to: ./myFlickr30k_resnet101/predictions.txt
Testing DataLoader 0: 100%|██████████| 16/16 [02:46<00:00,  0.10it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_loss         │     2.318966865539551     │
└───────────────────────────┴───────────────────────────┘

real    66m21.179s
user    73m9.119s
sys     1m5.035s
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ tree ./myFlickr30k_resnet101/
./myFlickr30k_resnet101/
├── detailed_predictions.json
├── final_model.ckpt
├── lightning_logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=19-step=1880.ckpt
│       ├── events.out.tfevents.1751451515.lst-hpc3090.2312626.0
│       ├── events.out.tfevents.1751455322.lst-hpc3090.2312626.1
│       └── hparams.yaml
├── predictions.txt
├── test_loss.png
└── train_val_loss.png

4 directories, 9 files
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ ll -lh ./myFlickr30k_resnet101/
total 493M
drwxrwxr-x  3 ye ye 4.0K Jul  2 18:24 ./
drwxrwxr-x 15 ye ye 4.0K Jul  2 17:18 ../
-rw-rw-r--  1 ye ye 4.9M Jul  2 18:24 detailed_predictions.json
-rw-------  1 ye ye 488M Jul  2 18:22 final_model.ckpt
drwxrwxr-x  3 ye ye 4.0K Jul  2 17:18 lightning_logs/
-rw-rw-r--  1 ye ye 149K Jul  2 18:24 predictions.txt
-rw-rw-r--  1 ye ye  42K Jul  2 18:24 test_loss.png
-rw-rw-r--  1 ye ye  32K Jul  2 18:24 train_val_loss.png
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

```
Testing DataLoader 0:  88%|████████▊ | 14/16 [02:01<00:17,  0.12it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0:  94%|█████████▍| 15/16 [02:10<00:08,  0.12it/s]Encoder output features before pooling: torch.Size([64, 2048, 7, 7])
Encoder output features after pooling: torch.Size([64, 2048, 1, 1])
Projection input dimension: 2048
Projection output dimension: 1024
Testing DataLoader 0: 100%|██████████| 16/16 [02:18<00:00,  0.12it/s]
TESTING COMPLETE - SUMMARY METRICS:
==================================================
BLEU-1: 0.4319
BLEU-2: 0.2485
BLEU-4: 0.1037
METEOR: 0.2955
ROUGE-L: 0.0030
CHRF++: 0.0310
Semantic_Similarity: 0.1868
Keyword_Overlap: 0.4885
Composite_Score: 0.1868
CIDEr: 0.0000
==================================================

Detailed results saved to: ./myFlickr30k_resnet152/detailed_predictions.json
Human-readable results saved to: ./myFlickr30k_resnet152/predictions.txt
Testing DataLoader 0: 100%|██████████| 16/16 [02:46<00:00,  0.10it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_loss         │     2.344592571258545     │
└───────────────────────────┴───────────────────────────┘

real    67m29.032s
user    74m8.205s
sys     1m5.015s
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ tree ./myFlickr30k_resnet152/
./myFlickr30k_resnet152/
├── detailed_predictions.json
├── final_model.ckpt
├── lightning_logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=19-step=1880.ckpt
│       ├── events.out.tfevents.1751465716.lst-hpc3090.2337709.0
│       ├── events.out.tfevents.1751469589.lst-hpc3090.2337709.1
│       └── hparams.yaml
├── predictions.txt
├── test_loss.png
└── train_val_loss.png

4 directories, 9 files
```
