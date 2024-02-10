def show(text, words, prediction):
    tag_indics = [i for i, element in enumerate(prediction) if element != 'O']
    # 创建一个HTML字符串
    html_string = "<p style='font-size: 20px; letter-spacing: 0px;'>"  # 这里将字体大小设置为20px
    html_string += "Original Text:    "+text + "<br>"
    html_string += "</p>"
    html_string += "<hr>"
    inserted_text = "Token & Tag:    "
    html_string += f'<span style="font-size: 20px;">{inserted_text}</span>'
    for i, word in enumerate(words):
        if prediction[i]!= "O":  
            html_string += f'<span style="border: 2px solid blue; padding: 3px; color: blue; display: inline-block;font-size: 22px;">{word}</span> '
        else:
            html_string += f'<span style="border: 1px solid black; padding: 3px; color: gray; display: inline-block;font-size: 20px;">{word}</span> '
    html_string += "<hr>"
    
    if len(tag_indics)>=1:
      for i in range(len(tag_indics)):
        event = 'Event' + str(i+1)
        html_string += f'<span style="font-size: 20px; color: green;">{event}</span> '
        trigger= "| Trigger word: "
        html_string += f'<span style="font-size: 20px;">{trigger}</span> '
        html_string += f'<span style="font-size: 20px; color: blue;">{words[tag_indics[i]]}</span> '
        etype = "| Event type: "
        html_string += f'<span style="font-size: 20px;">{etype}</span> '
        html_string += f'<span style="font-size: 20px; color: orange;">{prediction[tag_indics[i]][2:]}</span> '
        html_string += "<hr>"
    
    # 在Jupyter Notebook中展示HTML字符串
    return html_string