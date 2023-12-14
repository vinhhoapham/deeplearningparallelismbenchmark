from Tasks import CIFAR10, MNIST, FashionMNIST
from TrainingScheduler import Scheduler, Hogwild, DataParallelism, Greedy, Sequential, PipelineParallelism


def main():
    # Create a task
    tasks = {'CIFAR10': CIFAR10.task,
             'MNIST': MNIST.task,
            'FashionMNIST': FashionMNIST.task
        }
    scheduler_args = {
        'CIFAR10': {'num_processors': 8, 'batch_size': 128, 'epoch': 10},
        'MNIST': {'num_processors': 8, 'batch_size': 128, 'epoch': 10},
        'FashionMNIST': {'num_processors': 8, 'batch_size': 128, 'epoch': 10}
    }

    training_paradigms = {
        'Hogwild': Hogwild.master,
        'Greedy': Greedy.master,
    }

    for task_name, task in tasks.items():
        for training_paradigm_name, training_paradigm in training_paradigms.items():
            print(f"Running {task_name} - {training_paradigm_name}")
            scheduler = Scheduler.Scheduler(name=f"{task_name} - {training_paradigm_name}",
                                            train_paradigm=training_paradigm,
                                            task=task,
                                            **scheduler_args[task_name])
            scheduler.benchmark()




if __name__ == "__main__":
    main()



