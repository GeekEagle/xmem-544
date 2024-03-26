import tkinter as tk
from PIL import Image, ImageTk
# from embedding_demo import demo_main
# from inference.interact.fbrs_controller import FBRSController
# from model.network import XMem
# from inference.inference_core import InferenceCore
# from inference.interact.resource_manager import ResourceManager
# from inference.interact.interaction import *
# from inference.interact.gui_utils import *
# from inference.interact.interactive_utils import *
# from inference.interact.interactive_utils import color_map, index_numpy_to_one_hot_torch
# import torch
# import sys
# from argparse import ArgumentParser
import os

# config,model,fbrs,device

class ImagePlayerApp:
    def __init__(self, master,):
        self.master = master
        # self.config = args
        self.image_folder = './our_data/Frame/'
        self.mask_folder = "./our_data/Label/"
        self.image_files = sorted(os.listdir(self.image_folder))
        # self.mask_files = sorted(os.listdir(self.mask_folder))
        self.current_index = 0
        self.setup_gui()
        # self.num_objects = config['num_objects']
        # self.processor = InferenceCore(model,config)
        # self.processor.set_all_labels(list(range(1, self.num_objects + 1)))
        # self.fbrs_controller = fbrs
        # self.device = device

        # self.fbrs.toggled.connect(self.interaction_radio_clicked)

    def setup_gui(self,):
        self.canvas = tk.Canvas(self.master)
        self.canvas.grid(row=0, column=0, pady=20, sticky='w')
        # self.canvas.pack()

        self.imageLabel = tk.Label(self.master, text="Enter video name with format (e.g., blackswan): ")
        self.selected_option = tk.StringVar(self.master)
        self.selected_option.set(self.image_files[0])
        self.imagemenu = tk.OptionMenu(self.master, self.selected_option, *self.image_files)

        self.imageLabel.grid(row=1, column=0, pady=10, sticky='w')
        self.imagemenu.grid(row=1, column=1, pady=10)

        self.load_first_frame = tk.Button(self.master,text='see first frame',command=self.selected_video)
        self.load_first_frame.grid(row=2,column=1,pady=10)

        self.play_button1 = tk.Button(self.master, text="Play_forward", command=self.play_order_video)
        self.play_button1.grid(row=2, column=0, pady=10, sticky='w')

        self.play_button2 = tk.Button(self.master, text="Play_backward", command=self.play_reverse_video)
        self.play_button2.grid(row=3, column=0, pady=10, sticky='w')

        self.stop_button = tk.Button(self.master, text="Stop", command=self.stop_video)
        self.stop_button.grid(row=4, column=0, pady=10, sticky='w')

        self.canvas.bind("<Button-1>", self.mouse_click)
        self.photo = None  # 保存PhotoImage对象的引用
        self.playing = False

    def complete_interaction(self):
        if self.interaction is not None:
            self.clear_visualization()
            self.interaction = None

    def clear_visualization(self):
        self.vis_map.fill(0)
        self.vis_alpha.fill(0)

    def reset_this_interaction(self):
        self.complete_interaction()
        self.clear_visualization()
        self.interaction = None
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()

    def play_order_video(self):
        self.playing = True
        self.play_frame(1)

    def play_reverse_video(self):
        self.playing = True
        self.play_frame(-1)

    def pixel_pos_to_image_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_height, self.image_width
        nh, nw = self.canvas.size

        h_ratio = nh/oh
        w_ratio = nw/ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh/dominate_ratio, nw/dominate_ratio
        x -= (fw-ow)/2
        y -= (fh-oh)/2

        return x, y

    def get_scaled_pos(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        x = max(0, min(self.image_width-1, x))
        y = max(0, min(self.image_height-1, y))

        return x, y

    def is_pos_out_of_bound(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)
        out_of_bound = (
            (x < 0) or
            (y < 0) or
            (x > self.image_width-1) or
            (y > self.image_height-1)
        )
        return out_of_bound

    def mouse_click(self,event):
        if self.is_pos_out_of_bound(event.position().x(), event.position().y()):
            return

        h, w = self.image_height, self.image_width
        self.load_current_torch_image_mask()
        image = self.current_image_torch

        last_interaction = self.interaction
        new_interaction = None

        if (last_interaction is None or type(last_interaction) != ClickInteraction
                or last_interaction.tar_obj != self.current_object):
            self.complete_interaction()
            self.fbrs_controller.unanchor()
            new_interaction = ClickInteraction(image, self.current_prob, (h, w),
                                               self.fbrs_controller, self.current_object)
        print("Mouse clicked at", event.x, event.y)


    def selected_video(self):
        video_name = self.selected_option.get().strip()
        video_path = os.path.join(self.image_folder, video_name)
        frame_list = os.listdir(video_path)
        image_path = os.path.join(video_path, frame_list[self.current_index])
        image = Image.open(image_path)
        self.image_width, self.image_height = image.size
        self.canvas.config(width=self.image_width, height=self.image_height)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.master.update_idletasks()
        self.master.update()

    def play_frame(self,order):
        video_name = self.selected_option.get().strip()
        video_path = os.path.join(self.image_folder,video_name)
        frame_list = os.listdir(video_path)
        # mask_path = os.path.join(self.mask_folder,video_name)
        # mask_list = os.listdir(mask_path)
        # print(video_path)
        while self.playing and 0 <= self.current_index < len(video_path):
            image_path = os.path.join(video_path, frame_list[self.current_index])
            image = Image.open(image_path)
            # each_mask = os.path.join(mask_path, mask_list[self.current_index])
            # mask = Image.open(each_mask).convert("RGBA")
            # mask.putalpha(88)
            # image.paste(mask, (0, 0), mask)
            self.image_width, self.image_height = image.size
            self.canvas.config(width=self.image_width, height=self.image_height)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.master.update_idletasks()
            self.master.update()
            self.current_index += order
            self.master.after(100)
            if self.current_index == 0:
                self.current_index += 1
            elif self.current_index == len(self.image_files):
                self.current_index -= 1

    def stop_video(self):
        self.playing = False

def parameter():
    # Arguments parsing
    parser = ArgumentParser()
    parser.add_argument('--model', default='./saves/xmem_0126_110000.pth')
    # parser.add_argument('--s2m_model', default='saves/s2m.pth')
    parser.add_argument('--fbrs_model', default='saves/fbrs.pth')

    parser.add_argument('--buffer_size', help='Correlate with CPU memory consumption', type=int, default=10)

    parser.add_argument('--num_objects', type=int, default=1)

    # Long-memory options
    # Defaults. Some can be changed in the GUI.
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements',
                        help='LT_max in paper, increase if objects disappear for a long time',
                        type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', type=int, default=10)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int,
                        default=-1)
    parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
    parser.add_argument('--size', default=480, type=int,
                        help='Resize the shorter side to this size. -1 to use original resolution. ')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    #
    # args = parameter()
    # config = vars(args)

    # network = XMem(config, args.model, map_location=device).to(device).eval()
    # fbrs_controller = FBRSController(args.fbrs_model, device=device)


    root = tk.Tk()
    app = ImagePlayerApp(root)
    root.mainloop()
    # , config, network, fbrs_controller, device
