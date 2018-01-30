'''
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

import unittest
import random
import six
from builtins import super

from wc_sim.core.errors import SimulatorError
from wc_sim.core.event import Event
from wc_sim.core.simulation_object import EventQueue, SimulationObject, SimulationObjectInterface
from wc_sim.core.simulation_engine import SimulationEngine
from tests.core.some_message_types import InitMsg, Eg1, UnregisteredMsg
from wc_utils.util.misc import most_qual_cls_name

ALL_MESSAGE_TYPES = [InitMsg, Eg1]


class ExampleSimulationObject(SimulationObject, SimulationObjectInterface):

    def __init__(self, name):
        super().__init__(name)

    def send_initial_events(self, *args): pass

    def handler(self, event): pass

    @classmethod
    def register_subclass_handlers(this_class):
        SimulationObject.register_handlers(this_class,
            [(sim_msg_type, this_class.handler) for sim_msg_type in ALL_MESSAGE_TYPES])

    @classmethod
    def register_subclass_sent_messages(this_class):
        SimulationObject.register_sent_messages(this_class, ALL_MESSAGE_TYPES)


class TestEventQueue(unittest.TestCase):

    def setUp(self):
        self.event_queue = EventQueue()
        self.num_events = 5
        self.sender = sender = ExampleSimulationObject('sender')
        self.receiver = receiver = ExampleSimulationObject('receiver')
        for i in range(self.num_events):
            self.event_queue.schedule_event(i, i+1, sender, receiver, 'event type')
        self.simulator = SimulationEngine()
        self.simulator.register_object_types([ExampleSimulationObject])

    def test_next_event_time(self):
        empty_event_queue = EventQueue()
        self.assertEqual(float('inf'), empty_event_queue.next_event_time())
        self.assertEqual(1, self.event_queue.next_event_time())


class TestSimulationObject(unittest.TestCase):

    def setUp(self):
        self.simulator = SimulationEngine()
        self.simulator.register_object_types([ExampleSimulationObject])
        self.o1 = ExampleSimulationObject('o1')
        self.o2 = ExampleSimulationObject('o2')
        self.simulator.load_objects([self.o1, self.o2])
        self.simulator.initialize()

    def test_send_events(self):
        times=[2.0, 1.0, 0.5]
        # test both send_event methods
        for send_method in [self.o1.send_event, self.o1.send_event_absolute]:

            for t in times:
                send_method(t, self.o2, Eg1)

            tmp = sorted(times)
            while self.o2.event_queue.next_event_time() < float('inf'):
                self.assertEqual(self.o2.event_queue.next_event_time(), tmp.pop(0))
                self.o2.event_queue.next_events(self.o2)
            self.assertEqual(self.o2.event_queue.next_events(self.o2), [])

    def test_event_time_ties(self):
        self.o1.send_event(0, self.o2, Eg1)
        self.o1.send_event(2, self.o2, InitMsg)

        num=10
        self.o1.send_event(1, self.o2, InitMsg)
        for i in range(num):
            if random.choice([True, False]):
                self.o1.send_event(1, self.o2, Eg1)
            else:
                self.o1.send_event(1, self.o2, InitMsg)

        self.assertEqual(self.o2.event_queue.next_event_time(), 0)
        event_list = self.o2.event_queue.next_events(self.o2)
        self.assertEqual(event_list[0].event_time, 0)

        self.assertEqual(self.o2.event_queue.next_event_time(), 1)
        event_list = self.o2.event_queue.next_events(self.o2)
        # all InitMsg messages come before any Eg1 message,
        # and at least 1 InitMsg message exists
        expected_type = InitMsg
        switched = False
        for event in event_list:
            if not switched and event.event_type == most_qual_cls_name(Eg1):
                expected_type = Eg1
            self.assertEqual(event.event_type, most_qual_cls_name(expected_type))

        self.assertEqual(self.o2.event_queue.next_event_time(), 2)

    def test_event_queue_to_str(self):
        rv = self.o1.event_queue_to_str()
        self.assertIn(self.o1.name, rv)

        times=[2.0, 1.0, 0.5]
        for time in times:
            self.o1.send_event(time, self.o1, Eg1)
        rv = self.o1.event_queue_to_str()
        self.assertEqual(len(rv.split('\n')), len(times)+2)
        for time in times:
            self.assertIn(str(time), rv)

    def test_exceptions(self):
        delay = -1.0
        with self.assertRaises(SimulatorError) as context:
            self.o1.send_event(delay, self.o2, Eg1)
        self.assertEqual(str(context.exception),
            "delay < 0 in send_event(): {}".format(str(delay)))

        event_time = -1
        with self.assertRaises(SimulatorError) as context:
            self.o1.send_event_absolute(event_time, self.o2, Eg1)
        six.assertRegex(self, str(context.exception),
            "event_time \(-1.*\) < current time \(0.*\) in send_event_absolute\(\)")

        eq = EventQueue()
        with self.assertRaises(SimulatorError) as context:
            eq.schedule_event(2, 1, None, None, '')
        self.assertEqual(str(context.exception),
            "receive_time < send_time in schedule_event(): {} < {}".format(1, 2))

        with self.assertRaises(SimulatorError) as context:
            self.o1.send_event_absolute(2, self.o2, UnregisteredMsg)
        self.assertEqual(str(context.exception),
            "'{}' simulation objects not registered to send '{}' messages".format(
                most_qual_cls_name(self.o1),
                most_qual_cls_name(UnregisteredMsg)))

        with self.assertRaises(SimulatorError) as context:
            self.o1.add(self.simulator)
        self.assertEqual(str(context.exception),
            "SimulationObject '{}' is already part of a simulator".format(self.o1.name))

        SimulationObject.register_sent_messages(ExampleSimulationObject, [UnregisteredMsg])
        with self.assertRaises(SimulatorError) as context:
            self.o1.send_event_absolute(2, self.o2, UnregisteredMsg)
        self.assertEqual(str(context.exception),
            "'{}' simulation objects not registered to receive '{}' messages".format(
                most_qual_cls_name(self.o2),
                most_qual_cls_name(UnregisteredMsg)))

        T = 'TEST'
        with self.assertRaises(SimulatorError) as context:
            SimulationObject.register_handlers(ExampleSimulationObject, [(UnregisteredMsg, T)])
        self.assertEqual(str(context.exception), "handler '{}' must be callable".format(T))

        with self.assertRaises(SimulatorError) as context:
            # register the same type multiple times
            SimulationObject.register_handlers(ExampleSimulationObject,
                [(UnregisteredMsg, lambda x: 0), (UnregisteredMsg, lambda x: 0)])
        self.assertEqual(str(context.exception), "message type '{}' appears repeatedly".format(
            most_qual_cls_name(UnregisteredMsg)))


class ExampleUnregisteredSimulationObject(SimulationObject, SimulationObjectInterface):

    def __init__(self, name):
        super().__init__(name)

    def send_initial_events(self, *args): pass

    def handler(self, event): pass

    @classmethod
    def register_subclass_handlers(this_class): pass

    @classmethod
    def register_subclass_sent_messages(this_class): pass


class TestUnregisteredSimulationObject(unittest.TestCase):

    def test_get_receiving_priorities_dict(self):
        so = ExampleUnregisteredSimulationObject('o1')
        with self.assertRaises(SimulatorError) as context:
            so.get_receiving_priorities_dict()
        self.assertIn(str(context.exception),
            "SimulationObject type '{}' must call register_handlers()".format(so.__class__.__name__))
