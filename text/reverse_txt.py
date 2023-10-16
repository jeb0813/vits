from utils import load_filepaths_and_text
filelists=['./filelists/ljs_audio_text_val_filelist.txt.cleaned',
              './filelists/ljs_audio_text_test_filelist.txt.cleaned']

for filelist in filelists:
    print("START:", filelist)

    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in range(len(filepaths_and_text)):
        original_text = filepaths_and_text[i][1]
        reversed_text = original_text[::-1]
        filepaths_and_text[i][1] = reversed_text


    with open(filelist[:-12]+ "_reverse" + filelist[-12:], "w", encoding="utf-8") as f:
        f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

