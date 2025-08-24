# from PIL import Image, ImageDraw

# # 1. 读取原始PNG图像
# original_image = Image.open('/home/tirgan/a/liu3044/Project/Group_Transformer_hyper_patch_tgrs/results/hu/dffn/hu_dffn_out.png')

from PIL import Image, ImageDraw

def draw_dashed_line(draw, start, end, fill="black", width=2, dash_length=10, space_length=20):
    """绘制一个虚线"""
    x1, y1 = start
    x2, y2 = end
    total_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
    num_dashes = int(total_length / (dash_length + space_length))
    
    for i in range(num_dashes):
        start_fraction = i / num_dashes
        end_fraction = (i+1) / num_dashes
        draw.line([(x1 + (x2-x1)*start_fraction, y1 + (y2-y1)*start_fraction),
                   (x1 + (x2-x1)*end_fraction, y1 + (y2-y1)*end_fraction)],
                   fill=fill, width=width)

def draw_dashed_rect(draw, bounding_box, fill="black", width=2, dash_length=10, space_length=5):
    """绘制一个虚线矩形边框"""
    left, upper, right, lower = bounding_box
    # 绘制矩形的四条边
    draw_dashed_line(draw, (left, upper), (right, upper), fill=fill, width=width, dash_length=dash_length, space_length=space_length)
    draw_dashed_line(draw, (left, lower), (right, lower), fill=fill, width=width, dash_length=dash_length, space_length=space_length)
    draw_dashed_line(draw, (left, upper), (left, lower), fill=fill, width=width, dash_length=dash_length, space_length=space_length)
    draw_dashed_line(draw, (right, upper), (right, lower), fill=fill, width=width, dash_length=dash_length, space_length=space_length)

# original_image = Image.open('/home/tirgan/a/liu3044/Project/Group_Transformer_hyper_patch_tgrs_hu/results/bot/group_transformer/bot_group_transformer_out.png')
# draw = ImageDraw.Draw(original_image)
# width, height = original_image.size
# # 指定区域：例如左上角的 100x100 像素
# left, upper, right, lower = width-360, height//2-170, width-260, height//2-90

# # 放大指定区域至原图的高度
# aspect_ratio = (right - left) / (lower - upper)
# new_width = int(aspect_ratio * original_image.height)
# cropped_img = original_image.crop((left, upper, right, lower))
# zoomed_img = cropped_img.resize((new_width, original_image.height))

# # 创建一个新图像
# margin = 20
# new_image_width = original_image.width + zoomed_img.width + margin
# new_image = Image.new('RGB', (new_image_width, original_image.height), color=(255, 255, 255))
# new_image.paste(original_image, (0, 0))
# new_image.paste(zoomed_img, (original_image.width + margin, 0))

# draw = ImageDraw.Draw(new_image)

# # 绘制虚线矩形边框
# draw_dashed_rect(draw, [left, upper, right, lower])
# draw_dashed_rect(draw, [original_image.width + margin, 0, new_image_width, original_image.height])

# # 从放大图像的两个顶点绘制连线到原始图像的对应区域的两个顶点
# draw_dashed_line(draw, (right, upper), (original_image.width + margin, 0))
# draw_dashed_line(draw, (right, lower), (original_image.width + margin, original_image.height))

# # 保存图像
# new_image.save('./zoom_area/bot_group_transformer_out.png')




original_image = Image.open('/home/tirgan/a/liu3044/Project/Group_Transformer_hyper_patch_tgrs_hu/results/bot/group_transformer/bot_group_transformer_0_out.png')  # Change to your image path
draw = ImageDraw.Draw(original_image)
width, height = original_image.size

# Specified region for zooming (you can adjust these values)
left, upper, right, lower = width//2 - 95, height // 2 - 50, width - 95, height // 2

# Zoom in on specified region, ensuring it doesn't exceed the original image width
aspect_ratio = (right - left) / (lower - upper)
zoom_height = int(width / aspect_ratio)  # Calculate the height of the zoomed image based on original image width
cropped_img = original_image.crop((left, upper, right, lower))
zoomed_img = cropped_img.resize((width, zoom_height))

# Create a new image with original image and zoomed image, ensuring the total height doesn't exceed double the original
margin = 20
new_image_height = min(height + zoom_height + margin, height * 2)
new_image = Image.new('RGB', (width, new_image_height), color=(255, 255, 255))
new_image.paste(original_image, (0, 0))
new_image.paste(zoomed_img, (0, height + margin))

draw = ImageDraw.Draw(new_image)

# Draw dashed rectangle around the zoom region in the original image
draw_dashed_rect(draw, [left, upper, right, lower])

# Draw dashed rectangle around the zoomed image area
draw_dashed_rect(draw, [0, height + margin, width, height + margin + zoom_height])

# Draw connecting dashed lines from the zoomed region corners to the corresponding area below
draw_dashed_line(draw, (left, lower), (0, height + margin))
draw_dashed_line(draw, (right, lower), (width, height + margin))
# Save the new image
new_image.save('./zoom_area/bot_group_transformer_8_out.png')  # Change to your desired path
