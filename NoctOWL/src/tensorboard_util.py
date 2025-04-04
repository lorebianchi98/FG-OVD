from torch.utils.tensorboard import SummaryWriter

def log_train(writer, losses, step):
    writer.add_scalar('train_loss_triplet', losses['loss_triplet'], step)
    writer.add_scalar('train_loss_bg', losses['loss_bg'], step)
    writer.add_scalar('train_loss_bbox', losses['loss_bbox'], step)
    writer.add_scalar('train_loss_giou', losses['loss_giou'], step)
    writer.add_scalar('train_loss_rank', losses['loss_rank'], step)
    
def log_validation(writer, val_metrics, step):
    writer.add_scalar('validation_mAP', val_metrics['map'], step)
    
def log_lvis(writer, step, lvis_metrics=None, map_per_image=None):
    if lvis_metrics:
        writer.add_scalar('lvis_validation_mAP', lvis_metrics['map'], step)
        if 'map_c' in lvis_metrics:
            writer.add_scalar('lvis_mAP_common', lvis_metrics['map_c'], step)
            writer.add_scalar('lvis_mAP_frequent', lvis_metrics['map_f'], step)
            writer.add_scalar('lvis_mAP_rare', lvis_metrics['map_r'], step)
    if map_per_image:
        writer.add_scalar('lvis_validation_mAP_per_image', map_per_image, step)
    