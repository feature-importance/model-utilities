import unittest

import torch
from torch import nn
from torchbearer import Trial
from torchbearer.callbacks import TorchScheduler

from model_utilities.training.schedule import parse_learning_rate_arg


def run_get_lrs(lr_str, epochs=100):
    init_lr, schedule = parse_learning_rate_arg(lr_str)

    X = torch.rand((1024, 1))
    y = X + 1 + torch.randn((1024, 1)) * 0.01
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    cbs = []
    if schedule is not None:
        cbs.append(schedule)

    trial = Trial(model, optimizer, nn.MSELoss(), metrics=["lr"], callbacks=cbs)
    trial.with_generators(train_generator=loader)
    hist = trial.run(epochs=epochs, verbose=0)
    return [h['lr'] for h in hist]


TESTCASES = [
    ("0.1", [0.1] * 5),
    ("0.1*0.2@every10", [0.1 * 0.2 ** j for j in range(10) for _ in range(10)]),
    ("0.1*0.2@every4B", [0.1 * 0.2 ** j for j in range(10) for _ in range(1)]),
    ("0.1*0.2@every2B", [0.1 * 0.2, 0.1 * 0.2 * 0.2 * 0.2, 0.1 * 0.2 * 0.2 * 0.2 * 0.2 * 0.2]),
    ("0.1*0.2@[10,30,80]", [0.1] * 10 + [0.1 * 0.2] * 20 + [0.1 * 0.2 * 0.2] * 50 + [0.1 * 0.2 * 0.2 * 0.2] * 20),
    ("0.1*0.2@[4,12,32]B", [0.1 * 0.2] * 2 + [0.1 * 0.2 * 0.2] * 5 + [0.1 * 0.2 * 0.2 * 0.2] * 2),
    ("0.01<@10;0.1<@20;0.01", [0.01] * 10 + [0.1] * 10 + [0.01] * 50),
    ("0.01<@10;0.1@every1<@20;0.1", [0.01] * 10 + [0.01 * 0.1 ** j for j in range(10)] + [0.1] * 50),
    ("0.1*linearWarmup0.5,2", [0.1 * 0.5, 0.075] + [0.1] * 10),
    ("0.1*linearWarmup0.5,4", [0.1 * 0.5, 0.0625, 0.075, 0.0875] + [0.1] * 10),
    ("0.1*linearWarmup0.5,4<@5;1.0", [0.1 * 0.5, 0.0625, 0.075, 0.0875, 0.1] + [1] * 10),
    ("0.1*linearWarmup0.5,4<@4;1.0", [0.1 * 0.5, 0.0625, 0.075, 0.0875] + [1] * 10),
    ("0.1*linearWarmup0.5,4<@6;1.0", [0.1 * 0.5, 0.0625, 0.075, 0.0875, 0.1, 0.1] + [1] * 10),
    ("0.1*cosineAnnealing0.001,10", [0.1, 0.09757729755661011, 0.0905463412215599, 0.07959536998847742,
                                     0.0657963412215599, 0.0505, 0.03520365877844011, 0.02140463001152259,
                                     0.010453658778440109, 0.0034227024433899004]),
    ("0.1*cosineAnnealing1e-3,10", [0.1, 0.09757729755661011, 0.0905463412215599, 0.07959536998847742,
                                    0.0657963412215599, 0.0505, 0.03520365877844011, 0.02140463001152259,
                                    0.010453658778440109, 0.0034227024433899004]),
    ("0.1*linearWarmup0.5,4<@4;0.1*cosineAnnealing1e-3,10", [0.1 * 0.5, 0.0625, 0.075, 0.0875] +
     [0.1, 0.09757729755661011, 0.0905463412215599, 0.07959536998847742, 0.0657963412215599, 0.0505,
      0.03520365877844011, 0.02140463001152259, 0.010453658778440109, 0.0034227024433899004])
]


class TestSchedule(unittest.TestCase):
    def assertListAlmostEqual(self, list1, list2, tol=7):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol)

    def test_schedule(self):
        for schedule, expected in TESTCASES:
            with self.subTest(schedule=schedule, expected=expected):
                actual = run_get_lrs(schedule, epochs=len(expected))
                self.assertListAlmostEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
