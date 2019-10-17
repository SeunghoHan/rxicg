import os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
import sys

root_path = '/archive/MyHome/Programs/git/my_research/XAI_TP/rxicg'

image_captioner_path = os.path.join(root_path, 'captioner')
relationship_detector_path = os.path.join(root_path, 'visual_relationship')
explanation_part_path = os.path.join(root_path, 'explanation_part')

test_dataset_dir = '/archive/MyHome/Programs/git/my_research/dataset/coco_2014/test2014/'
# test_dataset_dir = os.path.join(root_path, 'test_dataset')

sys.path.insert(0, image_captioner_path)
sys.path.insert(0, relationship_detector_path)
sys.path.insert(0, explanation_part_path)

from image_captioner import Image_Captioner
from relationship_detector import Visual_Relationship_Detector
from explanation import Explainer



class Demo():
    def __init__(self, selected_img_num):
        self.test_dataset_path = '/archive/MyHome/Programs/git/my_research/Image_Captioning/Demo/testset_for_demo'
        
        self.i_captioner = Image_Captioner()
        self.r_detector = Visual_Relationship_Detector()
        self.explainer = Explainer()
        
        self.img_name = selected_img_num
        self.img_path = os.path.join(self.test_dataset_path, "{}.jpg".format{self.img_num})
        
        
    def run_captioner(self):
        i_results = self.i_captioner.get_results(self.img_path, self.img_name)
        return i_results
        
        
        
    def run_relationship_detector(self):
        self.test_image_name = image_name
        r_results = self.r_detector.generate_relationships_not_combining_boxes(self.img_path, self.img_name)
        return r_results
        
        
        
    def run_explainer(self, caps, cls_boxes, cls_names):
        rich_captions, used_relations = self.explainer.explanation(doLogging=False, 
                                                                  test_file_name=self.img_name)
        
        e_results = self.explainer.generate_final_output(caps,
                                                         rich_captions, 
                                                         used_relations, 
                                                         cls_boxes, 
                                                         cls_names,
                                                         self.img_path,
                                                         self.img_name)
        e_results['rich_caps'] = rich_captions
        e_results['used_relations'] = used_relations
        
        return e_results